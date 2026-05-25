"""
VTAB-1K Fine-Tuning Benchmark — PEFT method implementations on HuggingFace ViT

Methods: Full Fine-Tune, Linear Probe, VPT-Deep, SSF, AdaptFormer, CFT
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
# Full Fine-Tune
# =============================================================================
def apply_full_finetune(model, num_classes, config):
    """All parameters trainable + new head."""
    return model


# =============================================================================
# Linear Probe
# =============================================================================
def apply_linear_probe(model, num_classes, config):
    """Only classifier head trainable."""
    freeze_backbone(model)
    return model


# =============================================================================
# VPT-Deep
# Ref: vpt-main/src/models/vit_prompt/vit.py
# Prepends learnable prompt tokens at every transformer layer.
# =============================================================================
def apply_vpt_deep(model, num_classes, config):
    """Apply VPT-Deep: learnable prompts prepended at every layer."""
    freeze_backbone(model)

    hidden_size  = model.config.hidden_size
    num_layers   = model.config.num_hidden_layers
    num_tokens   = config["vpt_num_tokens"]
    patch_size   = config["patch_size"]

    # Xavier-uniform init (same as VPT repo)
    val = math.sqrt(6.0 / float(3 * reduce(mul, (patch_size, patch_size), 1) + hidden_size))

    prompt_embeddings = nn.ParameterList()
    for _ in range(num_layers):
        p = nn.Parameter(torch.zeros(1, num_tokens, hidden_size))
        nn.init.uniform_(p.data, -val, val)
        prompt_embeddings.append(p)

    prompt_dropout = nn.Dropout(config["vpt_dropout"])

    model.vpt_prompt_embeddings = prompt_embeddings
    model.vpt_prompt_dropout = prompt_dropout
    model._vpt_num_tokens = num_tokens

    encoder = model.vit.encoder

    def vpt_encoder_forward(
        hidden_states,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        B = hidden_states.shape[0]

        for i, layer_module in enumerate(encoder.layer):
            prompt_tokens = prompt_embeddings[i].expand(B, -1, -1)
            hidden_states = torch.cat(
                [hidden_states[:, :1, :], prompt_tokens, hidden_states[:, 1:, :]],
                dim=1,
            )

            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs if isinstance(layer_outputs, torch.Tensor) else layer_outputs[0]

            # Strip prompts between layers, NOT after the last one
            if i < len(encoder.layer) - 1:
                hidden_states = torch.cat(
                    [hidden_states[:, :1, :], hidden_states[:, 1 + num_tokens:, :]],
                    dim=1,
                )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    encoder.forward = vpt_encoder_forward
    return model


# =============================================================================
# SSF — Scale & Shift Feature adjustment
# Ref: SSF-main/models/vision_transformer.py
# =============================================================================
class SSFWrapper(nn.Module):
    def __init__(self, original_module, dim):
        super().__init__()
        self.original = original_module
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        nn.init.trunc_normal_(self.scale, std=0.02)
        self.scale.data += 1.0
        nn.init.trunc_normal_(self.shift, std=0.02)

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    def forward(self, *args, **kwargs):
        x = self.original(*args, **kwargs)
        if isinstance(x, tuple):
            return (self._apply_ssf(x[0]),) + x[1:]
        return self._apply_ssf(x)

    def _apply_ssf(self, x):
        if x.dim() == 4:  # Conv2d output: [B, C, H, W]
            return x * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)
        return x * self.scale + self.shift


def apply_ssf(model, num_classes, config):
    """Apply SSF: learnable scale & shift after every LayerNorm / Linear."""
    freeze_backbone(model)
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size

    # Patch embed
    model.vit.embeddings.patch_embeddings.projection = SSFWrapper(
        model.vit.embeddings.patch_embeddings.projection, hidden_size)

    for i, layer in enumerate(model.vit.encoder.layer):
        layer.layernorm_before = SSFWrapper(layer.layernorm_before, hidden_size)
        layer.attention.attention.query = SSFWrapper(layer.attention.attention.query, hidden_size)
        layer.attention.attention.key = SSFWrapper(layer.attention.attention.key, hidden_size)
        layer.attention.attention.value = SSFWrapper(layer.attention.attention.value, hidden_size)
        layer.attention.output.dense = SSFWrapper(layer.attention.output.dense, hidden_size)
        layer.layernorm_after = SSFWrapper(layer.layernorm_after, hidden_size)
        layer.intermediate = SSFWrapper(layer.intermediate, intermediate_size)
        layer.output.dense = SSFWrapper(layer.output.dense, hidden_size)

    model.vit.layernorm = SSFWrapper(model.vit.layernorm, hidden_size)
    return model


# =============================================================================
# AdaptFormer — Parallel adapter in FFN blocks
# Ref: AdaptFormer-main/models/adapter.py
# =============================================================================
class Adapter(nn.Module):
    """Adapter module — exact replica from AdaptFormer repo."""

    def __init__(self, d_model, bottleneck, dropout=0.1,
                 adapter_scalar="1.0", adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ("in", "out"):
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            return up + residual
        return up


def apply_adaptformer(model, num_classes, config):
    """Apply AdaptFormer: parallel adapter in each FFN block."""
    freeze_backbone(model)

    hidden_size = model.config.hidden_size
    bottleneck  = config["adapter_bottleneck"]
    dropout     = config["adapter_dropout"]
    scalar      = config["adapter_scalar"]

    adapters = nn.ModuleList()

    for i, layer in enumerate(model.vit.encoder.layer):
        adapter = Adapter(
            d_model=hidden_size,
            bottleneck=bottleneck,
            dropout=dropout,
            adapter_scalar=scalar,
            adapter_layernorm_option="in",
        )
        adapters.append(adapter)

        def make_adapter_forward(layer_ref, adapt):
            def new_forward(hidden_states, head_mask=None, output_attentions=False):
                self_attention_outputs = layer_ref.attention(
                    layer_ref.layernorm_before(hidden_states),
                )
                if isinstance(self_attention_outputs, torch.Tensor):
                    attention_output = self_attention_outputs
                    outputs = ()
                else:
                    attention_output = self_attention_outputs[0]
                    outputs = self_attention_outputs[1:]
                hidden_states = attention_output + hidden_states

                adapt_x = adapt(hidden_states, add_residual=False)

                layer_output = layer_ref.layernorm_after(hidden_states)
                layer_output = layer_ref.intermediate(layer_output)
                layer_output = layer_ref.output(layer_output, hidden_states)
                layer_output = layer_output + adapt_x

                return (layer_output,) + outputs if outputs else layer_output
            return new_forward

        layer.adapter = adapter
        layer.forward = make_adapter_forward(layer, adapter)

    model.adapters = adapters
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(model.config.hidden_size, affine=False, eps=1e-6),
        model.classifier,
    )
    return model


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

    if method == "full_finetune":
        model = apply_full_finetune(model, num_classes, config)
    elif method == "linear_probe":
        model = apply_linear_probe(model, num_classes, config)
    elif method == "vpt_deep":
        model = apply_vpt_deep(model, num_classes, config)
    elif method == "ssf":
        model = apply_ssf(model, num_classes, config)
    elif method == "adaptformer":
        model = apply_adaptformer(model, num_classes, config)
    elif method == "cft":
        model = apply_cft(model, num_classes, config, selected_nodes, nodes_map)
    else:
        raise ValueError(f"Unknown method: {method}")

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
