"""
VTAB-1K Fine-Tuning Benchmark — PEFT method implementations on HuggingFace ViT
"""
import math
from collections import defaultdict
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ViTForImageClassification
from transformers.modeling_outputs import BaseModelOutput


# =============================================================================
# Helpers
# =============================================================================
def freeze_backbone(model):
    """Freeze all parameters except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())



# =============================================================================
# CFT — Circuit Fine-Tuning
# =============================================================================
def apply_cft(model, num_classes, config, selected_nodes=None, nodes_map=None):
    """Apply CFT: freeze everything, then unfreeze only circuit nodes + head.
    Uses gradient masking for head-level selectivity within shared Q/K/V/O matrices.
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Always-on: patch+pos embeddings + all LayerNorms (~0.9% of ViT-B).
    for param in model.vit.embeddings.parameters():
        param.requires_grad = True
    for layer in model.vit.encoder.layer:
        for param in layer.layernorm_before.parameters():
            param.requires_grad = True
        for param in layer.layernorm_after.parameters():
            param.requires_grad = True
    for param in model.vit.layernorm.parameters():
        param.requires_grad = True

    if selected_nodes is None:
        return model
    if nodes_map is None:
        raise ValueError("apply_cft: nodes_map must be provided when selected_nodes is not None")

    hidden_size = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    d_head = hidden_size // n_heads
    num_layers = len(model.vit.encoder.layer)

    selected_heads_per_layer = defaultdict(set)
    selected_mlps = set()
    selected_nodes = list(selected_nodes)
    for node_name in selected_nodes:
        if node_name not in nodes_map:
            raise ValueError(f"apply_cft: unknown node '{node_name}' (not found in nodes_map)")
        info = nodes_map[node_name]
        node_type = info.get("type")
        layer_idx = info.get("layer_idx")
        if not isinstance(layer_idx, int) or layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(
                f"apply_cft: invalid layer_idx for node '{node_name}': {layer_idx}"
            )
        if node_type == "head":
            head_idx = info.get("head_idx")
            if not isinstance(head_idx, int) or head_idx < 0 or head_idx >= n_heads:
                raise ValueError(
                    f"apply_cft: invalid head_idx for node '{node_name}': {head_idx}"
                )
            selected_heads_per_layer[layer_idx].add(head_idx)
        elif node_type == "mlp":
            selected_mlps.add(layer_idx)
        else:
            raise ValueError(
                f"apply_cft: unsupported node type '{node_type}' for node '{node_name}'"
            )

    model._cft_grad_hooks = []
    model._cft_no_weight_decay_params = []

    for layer_idx, head_set in selected_heads_per_layer.items():
        layer = model.vit.encoder.layer[layer_idx]

        mask = torch.zeros(hidden_size)
        for h in head_set:
            mask[h * d_head : (h + 1) * d_head] = 1.0
        layer.register_buffer("_cft_head_mask", mask)

        for proj in [layer.attention.attention.query,
                     layer.attention.attention.key,
                     layer.attention.attention.value]:
            proj.weight.requires_grad = True
            model._cft_no_weight_decay_params.append(proj.weight)
            if proj.bias is not None:
                proj.bias.requires_grad = True
                model._cft_no_weight_decay_params.append(proj.bias)
            model._cft_grad_hooks.append(
                proj.weight.register_hook(
                    lambda g, layer_ref=layer: g * layer_ref._cft_head_mask.unsqueeze(1)
                )
            )
            if proj.bias is not None:
                model._cft_grad_hooks.append(
                    proj.bias.register_hook(
                        lambda g, layer_ref=layer: g * layer_ref._cft_head_mask
                    )
                )

        o_proj = layer.attention.output.dense
        o_proj.weight.requires_grad = True
        model._cft_no_weight_decay_params.append(o_proj.weight)
        model._cft_grad_hooks.append(
            o_proj.weight.register_hook(
                lambda g, layer_ref=layer: g * layer_ref._cft_head_mask.unsqueeze(0)
            )
        )
        if o_proj.bias is not None:
            o_proj.bias.requires_grad = True
            model._cft_no_weight_decay_params.append(o_proj.bias)
            model._cft_grad_hooks.append(
                o_proj.bias.register_hook(
                    lambda g, layer_ref=layer: g * layer_ref._cft_head_mask
                )
            )

    for layer_idx in selected_mlps:
        layer = model.vit.encoder.layer[layer_idx]
        for param in layer.intermediate.parameters():
            param.requires_grad = True
        for param in layer.output.dense.parameters():
            param.requires_grad = True

    effective_params = 0
    for node_name in selected_nodes:
        effective_params += nodes_map[node_name]["param_count"]
    effective_params += sum(p.numel() for p in model.classifier.parameters())
    model._cft_effective_params = effective_params

    return model


# =============================================================================
# Unified model builder
# =============================================================================
def build_model(method, num_classes, config, device=None,
                selected_nodes=None, nodes_map=None):
    """Factory: load pretrained ViT and apply specified PEFT method."""
    model = ViTForImageClassification.from_pretrained(config["model_name"])
    model.classifier = nn.Linear(model.config.hidden_size, num_classes)
    nn.init.normal_(model.classifier.weight, std=1e-5)
    nn.init.zeros_(model.classifier.bias)
    head_drop = config.get("head_dropout", 0.0)
    if head_drop > 0:
        model.classifier = nn.Sequential(nn.Dropout(head_drop), model.classifier)


    model = apply_cft(model, num_classes, config, selected_nodes, nodes_map)


    if device is not None:
        model = model.to(device)

    trainable = count_trainable_params(model)
    total = count_total_params(model)
    if method == "cft" and hasattr(model, '_cft_effective_params'):
        effective = model._cft_effective_params
        print(f"  [{method}] Trainable: {trainable:,} (effective after masking: {effective:,}, {100*effective/total:.2f}%)")
    else:
        print(f"  [{method}] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model
