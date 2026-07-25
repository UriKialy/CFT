"""
Swin PEFT methods

CFT-only build keeps all original methods (full_finetune, linear_probe, vpt_deep,
ssf, adaptformer, cft) so build_model still works, but only cft is used in this
repo's CLI. To use other methods, call build_model(method=...) directly.
"""
import math
from collections import defaultdict
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Swinv2ForImageClassification, Swinv2Config, Swinv2Model
from transformers.modeling_outputs import BaseModelOutput



from transformers.modeling_outputs import BaseModelOutput
from collections import defaultdict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def freeze_backbone(model):
    """Freeze all parameters except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


# ======================= CFT ===============================================
def apply_cft(model, num_classes, config, selected_nodes=None, nodes_map=None, task_name=""):
    """Apply CFT: freeze everything, then unfreeze only circuit nodes + head + layerNorm.
    Uses gradient masking for head-level selectivity within shared Q/K/V/O matrices.
    Adapted for HF SwinV2 hierarchical structure.
    """
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    if selected_nodes is None:
        return model

    embed_dim = model.config.embed_dim
    num_heads_per_stage = model.config.num_heads  # [4, 8, 16, 32]

    # Group selections by (stage_idx, block_idx)
    selected_heads_per_block = defaultdict(set)  # (stage, block) → {head_idx, ...}
    selected_mlps = set()                         # (stage, block)
    for node_name in selected_nodes:
        info = nodes_map[node_name]
        if info["type"] == "head":
            selected_heads_per_block[(info["stage_idx"], info["block_idx"])].add(info["head_idx"])
        elif info["type"] == "mlp":
            selected_mlps.add((info["stage_idx"], info["block_idx"]))

    swinv2 = model.swinv2
    model._cft_grad_hooks = []

    # ── Unfreeze attention layers with gradient masks ──
    for (stage_idx, block_idx), head_set in selected_heads_per_block.items():
        stage_dim = embed_dim * (2 ** stage_idx)
        n_heads = num_heads_per_stage[stage_idx]
        d_head = stage_dim // n_heads
        block = swinv2.encoder.layers[stage_idx].blocks[block_idx]

        mask = torch.zeros(stage_dim, device="cpu")
        for h in head_set:
            mask[h * d_head : (h + 1) * d_head] = 1.0

        # Q, K, V — mask rows
        for proj in [block.attention.self.query,
                     block.attention.self.key,
                     block.attention.self.value]:
            proj.weight.requires_grad = True
            if proj.bias is not None:
                proj.bias.requires_grad = True
            m = mask.clone()
            model._cft_grad_hooks.append(
                proj.weight.register_hook(lambda g, m=m: g * m.to(g.device).unsqueeze(1)))
            if proj.bias is not None:
                model._cft_grad_hooks.append(
                    proj.bias.register_hook(lambda g, m=m: g * m.to(g.device)))

        # Output dense — mask columns
        o_proj = block.attention.output.dense
        o_proj.weight.requires_grad = True
        if o_proj.bias is not None:
            o_proj.bias.requires_grad = True
        m = mask.clone()
        model._cft_grad_hooks.append(
            o_proj.weight.register_hook(lambda g, m=m: g * m.to(g.device).unsqueeze(0)))

    # ── Unfreeze MLP layers ──
    for (stage_idx, block_idx) in selected_mlps:
        block = swinv2.encoder.layers[stage_idx].blocks[block_idx]
        for param in block.intermediate.parameters():
            param.requires_grad = True
        for param in block.output.dense.parameters():
            param.requires_grad = True

    # ── Unfreeze all LayerNorm params (~0.045% of backbone) ──
    for name, param in model.named_parameters():
        if "layernorm" in name.lower() or "layer_norm" in name.lower():
            param.requires_grad = True

    # ── Dropout for unfrozen layers ──
    dropout_rate = CFT_DROPOUT.get(task_name, 0.0)
    if dropout_rate > 0:
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate

    # ── Effective params ──
    effective_params = 0
    for node_name in selected_nodes:
        effective_params += nodes_map[node_name]["param_count"]
    effective_params += sum(p.numel() for p in model.classifier.parameters())
    model._cft_effective_params = effective_params

    # ── No-weight-decay params (partially masked Q/K/V/O) ──
    no_wd_params = []
    for (stage_idx, block_idx), head_set in selected_heads_per_block.items():
        n_heads = num_heads_per_stage[stage_idx]
        if len(head_set) == n_heads:
            continue  # All heads selected — normal wd
        block = swinv2.encoder.layers[stage_idx].blocks[block_idx]
        for proj in [block.attention.self.query,
                     block.attention.self.key,
                     block.attention.self.value,
                     block.attention.output.dense]:
            no_wd_params.append(proj.weight)
            if proj.bias is not None:
                no_wd_params.append(proj.bias)
    model._cft_no_weight_decay_params = no_wd_params

    return model

# ======================= UNIFIED BUILD =====================================
def build_model(method, num_classes, config, selected_nodes=None, nodes_map=None, task_name=""):
    """Factory: load pretrained SwinV2 and apply specified PEFT method."""
    model = Swinv2ForImageClassification.from_pretrained(config["model_name"])

    # SwinV2 classifier: final dim = embed_dim * 2^(num_stages-1)
    final_dim = model.config.embed_dim * (2 ** (len(model.config.depths) - 1))  # 1024
    model.classifier = nn.Linear(final_dim, num_classes)
    nn.init.normal_(model.classifier.weight, std=1e-5)
    nn.init.zeros_(model.classifier.bias)

    model = apply_cft(model, num_classes, config, selected_nodes, nodes_map, task_name)

    model = model.to(device)
    trainable = count_trainable_params(model)
    total = count_total_params(model)
    if method == "cft" and hasattr(model, '_cft_effective_params'):
        effective = model._cft_effective_params
        print(f"  [{method}] Trainable: {trainable:,} (effective after masking: {effective:,}, {100*effective/total:.2f}%)")
    else:
        print(f"  [{method}] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model

# Quick sanity check
