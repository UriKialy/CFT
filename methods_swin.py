"""
Swin PEFT methods — extracted verbatim from Swin_vtab1k_CFT.ipynb cell 7.

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

# =============================================================================
# CELL 5: PEFT Method Implementations for HuggingFace SwinV2
# =============================================================================
# Each method modifies a base Swinv2ForImageClassification in-place.
# VPT:  follows vpt-main/src/models/vit_prompt/swin_transformer.py
# SSF:  follows SSF-main/models/swin_transformer.py
# AdaptFormer: follows AdaptFormer-main/models/adapter.py
# HF SwinV2 uses POST-NORM (residual-post-norm) + cosine attention +
# continuous position bias MLP + separate Q/K/V.
# =============================================================================

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


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


# ======================= FULL FINE-TUNE ====================================
def apply_full_finetune(model, num_classes, config):
    """All parameters trainable + new head."""
    return model


# ======================= LINEAR PROBE ======================================
def apply_linear_probe(model, num_classes, config):
    """Only classifier head trainable."""
    freeze_backbone(model)
    return model


# ======================= VPT-DEEP ==========================================
# Ref: vpt-main/src/models/vit_prompt/swin_transformer.py
# Prompts are prepended to sequence, broadcast into EVERY local window for
# attention, then stripped from windows and mean-pooled across windows.
# Deep prompts: per-stage dims, REPLACED between blocks.
# PatchMerging: strip→merge spatial→upsample prompts (4×concat)→re-prepend.
# ---------------------------------------------------------------------------

def apply_vpt_deep(model, num_classes, config):
    """Apply VPT-Deep: learnable prompts injected into Swin windows."""

    freeze_backbone(model)

    embed_dim  = model.config.embed_dim       # 128
    depths     = model.config.depths          # [2, 2, 18, 2]
    patch_size = config["patch_size"]
    num_tokens = config["vpt_num_tokens"]

    # ── Prompt Embeddings: per-stage dims ──
    # Repo: deep_prompt_embeddings_0 has depths[0]-1 entries (first block uses initial prompt)
    #        deep_prompt_embeddings_{1,2,3} have depths[i] entries
    val = math.sqrt(6.0 / float(3 * reduce(mul, (patch_size, patch_size), 1) + embed_dim))

    # Initial prompt (stage 0 embed_dim)
    initial_prompt = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
    nn.init.uniform_(initial_prompt.data, -val, val)

    # Deep prompts per stage
    deep_prompts = nn.ParameterList()
    for stage_idx, depth in enumerate(depths):
        stage_dim = embed_dim * (2 ** stage_idx)
        val_s = math.sqrt(6.0 / float(3 * reduce(mul, (patch_size, patch_size), 1) + stage_dim))
        if stage_idx == 0:
            # First block uses initial_prompt, remaining depth-1 blocks get deep prompts
            p = nn.Parameter(torch.zeros(depth - 1, num_tokens, stage_dim))
        else:
            p = nn.Parameter(torch.zeros(depth, num_tokens, stage_dim))
        nn.init.uniform_(p.data, -val_s, val_s)
        deep_prompts.append(p)

    prompt_dropout = nn.Dropout(config["vpt_dropout"])

    # Register on model
    model.vpt_initial_prompt = initial_prompt
    model.vpt_deep_prompts = deep_prompts
    model.vpt_prompt_dropout = prompt_dropout
    model._vpt_num_tokens = num_tokens

    swinv2 = model.swinv2

    # ── Patch Swinv2SelfAttention.forward: pad position bias & mask for prompts ──
    def make_patched_attention_forward(orig_attn, np_tokens):
        orig_forward = orig_attn.forward

        def patched_attn_forward(hidden_states, attention_mask=None, output_attentions=False):
            # hidden_states: (nW*B, np + ws*ws, C) — prompts prepended to each window
            has_prompts = hidden_states.shape[1] > orig_attn.window_size[0] * orig_attn.window_size[1]

            if not has_prompts:
                return orig_forward(hidden_states, attention_mask, output_attentions)

            np_ = np_tokens
            batch_size, seq_len, num_channels = hidden_states.shape
            ws_sq = seq_len - np_

            # Separate Q/K/V projections
            query_layer = (
                orig_attn.query(hidden_states)
                .view(batch_size, -1, orig_attn.num_attention_heads, orig_attn.attention_head_size)
                .transpose(1, 2)
            )
            key_layer = (
                orig_attn.key(hidden_states)
                .view(batch_size, -1, orig_attn.num_attention_heads, orig_attn.attention_head_size)
                .transpose(1, 2)
            )
            value_layer = (
                orig_attn.value(hidden_states)
                .view(batch_size, -1, orig_attn.num_attention_heads, orig_attn.attention_head_size)
                .transpose(1, 2)
            )

            # Cosine attention
            attention_scores = nn.functional.normalize(query_layer, dim=-1) @ nn.functional.normalize(
                key_layer, dim=-1
            ).transpose(-2, -1)
            logit_scale = torch.clamp(orig_attn.logit_scale, max=math.log(1.0 / 0.01)).exp()
            attention_scores = attention_scores * logit_scale

            # Position bias: original is [nH, ws*ws, ws*ws], pad to [nH, np+ws*ws, np+ws*ws]
            relative_position_bias_table = orig_attn.continuous_position_bias_mlp(
                orig_attn.relative_coords_table
            ).view(-1, orig_attn.num_attention_heads)
            relative_position_bias = relative_position_bias_table[
                orig_attn.relative_position_index.view(-1)
            ].view(
                orig_attn.window_size[0] * orig_attn.window_size[1],
                orig_attn.window_size[0] * orig_attn.window_size[1],
                -1,
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

            # Pad: zeros for prompt↔prompt and prompt↔spatial interactions
            nH = orig_attn.num_attention_heads
            dev = attention_scores.device
            dtype = attention_scores.dtype
            # [nH, np_, ws_sq]
            pad_top = torch.zeros(nH, np_, ws_sq, device=dev, dtype=dtype)
            # [nH, np_+ws_sq, np_]
            pad_left = torch.zeros(nH, np_ + ws_sq, np_, device=dev, dtype=dtype)
            # Stack: [nH, np_+ws_sq, np_+ws_sq]
            padded_bias = torch.cat([pad_top, relative_position_bias], dim=1)  # [nH, np_+ws_sq, ws_sq]
            padded_bias = torch.cat([pad_left, padded_bias], dim=2)  # [nH, np_+ws_sq, np_+ws_sq]

            attention_scores = attention_scores + padded_bias.unsqueeze(0)

            # Attention mask padding
            if attention_mask is not None:
                mask_shape = attention_mask.shape[0]  # nW
                nW_B = batch_size
                B_actual = nW_B // mask_shape

                # Pad mask: [nW, ws_sq, ws_sq] → [nW, np_+ws_sq, np_+ws_sq]
                _nW, _h, _w = attention_mask.shape
                mask_pad_top = torch.zeros(_nW, np_, _w, device=dev, dtype=dtype)
                padded_mask = torch.cat([mask_pad_top, attention_mask], dim=1)
                mask_pad_left = torch.zeros(_nW, np_ + _h, np_, device=dev, dtype=dtype)
                padded_mask = torch.cat([mask_pad_left, padded_mask], dim=2)

                attention_scores = attention_scores.view(
                    B_actual, mask_shape, orig_attn.num_attention_heads, seq_len, seq_len
                ) + padded_mask.unsqueeze(1).unsqueeze(0)
                attention_scores = attention_scores.view(-1, orig_attn.num_attention_heads, seq_len, seq_len)

            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = orig_attn.dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_shape = context_layer.size()[:-2] + (orig_attn.all_head_size,)
            context_layer = context_layer.view(new_shape)

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            return outputs

        return patched_attn_forward

    # ── Patch Swinv2Layer.forward: inject prompts into windows ──
    def make_patched_block_forward(block, np_tokens):
        orig_maybe_pad = block.maybe_pad
        orig_get_attn_mask = block.get_attn_mask

        def patched_block_forward(hidden_states, input_dimensions, output_attentions=False):
            height, width = input_dimensions
            batch_size, full_len, channels = hidden_states.size()

            # ── Strip prompts ──
            prompt_emb = hidden_states[:, :np_tokens, :]
            spatial = hidden_states[:, np_tokens:, :]

            shortcut_spatial = spatial
            shortcut_prompt = prompt_emb

            # ── Spatial processing (identical to original) ──
            spatial = spatial.view(batch_size, height, width, channels)
            spatial, pad_values = orig_maybe_pad(spatial, height, width)
            _, height_pad, width_pad, _ = spatial.shape

            if block.shift_size > 0:
                shifted = torch.roll(spatial, shifts=(-block.shift_size, -block.shift_size), dims=(1, 2))
            else:
                shifted = spatial

            # Window partition
            from transformers.models.swinv2.modeling_swinv2 import window_partition, window_reverse
            windows = window_partition(shifted, block.window_size)  # (nW*B, ws, ws, C)
            windows = windows.view(-1, block.window_size * block.window_size, channels)
            num_windows = windows.shape[0] // batch_size

            # ── Broadcast prompts into every window (VPT repo logic) ──
            prompt_expanded = prompt_emb.unsqueeze(0).expand(num_windows, -1, -1, -1)
            prompt_expanded = prompt_expanded.reshape(-1, np_tokens, channels)  # (nW*B, np, C)
            windows_with_prompts = torch.cat([prompt_expanded, windows], dim=1)  # (nW*B, np+ws*ws, C)

            # Attention mask
            attn_mask = orig_get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
            if attn_mask is not None:
                attn_mask = attn_mask.to(windows.device)

            # ── Attention (patched to handle prompts) ──
            attention_outputs = block.attention(windows_with_prompts, attn_mask, output_attentions)
            attention_output = attention_outputs[0]  # (nW*B, np+ws*ws, C)

            # ── Strip prompts from windows, mean-pool across windows (VPT repo) ──
            prompt_from_windows = attention_output[:, :np_tokens, :]  # (nW*B, np, C)
            attn_spatial = attention_output[:, np_tokens:, :]  # (nW*B, ws*ws, C)

            # Mean-pool prompts across windows → (B, np, C)
            prompt_from_windows = prompt_from_windows.view(num_windows, batch_size, np_tokens, channels)
            prompt_emb = prompt_from_windows.mean(0)  # (B, np, C)

            # ── Window reverse ──
            attn_spatial = attn_spatial.view(-1, block.window_size, block.window_size, channels)
            shifted_back = window_reverse(attn_spatial, block.window_size, height_pad, width_pad)

            if block.shift_size > 0:
                spatial_out = torch.roll(shifted_back, shifts=(block.shift_size, block.shift_size), dims=(1, 2))
            else:
                spatial_out = shifted_back

            was_padded = pad_values[3] > 0 or pad_values[5] > 0
            if was_padded:
                spatial_out = spatial_out[:, :height, :width, :].contiguous()

            spatial_out = spatial_out.view(batch_size, height * width, channels)

            # ── Re-combine and apply POST-NORM residual ──
            attn_out = torch.cat([prompt_emb, spatial_out], dim=1)  # (B, np+H*W, C)
            shortcut = torch.cat([shortcut_prompt, shortcut_spatial], dim=1)
            hidden_states = shortcut + block.drop_path(block.layernorm_before(attn_out))

            # ── MLP (prompts go through MLP too, same as VPT repo) ──
            layer_output = block.intermediate(hidden_states)
            layer_output = block.output(layer_output)
            layer_output = hidden_states + block.drop_path(block.layernorm_after(layer_output))

            outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
            return outputs

        return patched_block_forward

    # ── Patch Swinv2PatchMerging.forward: strip→merge→upsample prompts ──
    def make_patched_downsample_forward(downsample, np_tokens):
        orig_maybe_pad = downsample.maybe_pad

        def patched_downsample_forward(input_feature, input_dimensions):
            height, width = input_dimensions
            batch_size, full_len, num_channels = input_feature.shape

            # ── Strip prompts ──
            prompt_emb = input_feature[:, :np_tokens, :]  # (B, np, C)
            spatial = input_feature[:, np_tokens:, :]  # (B, H*W, C)

            # ── Upsample prompts: 4×concat (VPT repo: PromptedPatchMerging) ──
            prompt_emb = torch.cat([prompt_emb, prompt_emb, prompt_emb, prompt_emb], dim=-1)  # (B, np, 4*C)

            # ── Spatial merge (standard PatchMerging) ──
            spatial = spatial.view(batch_size, height, width, num_channels)
            spatial = orig_maybe_pad(spatial, height, width)
            f0 = spatial[:, 0::2, 0::2, :]
            f1 = spatial[:, 1::2, 0::2, :]
            f2 = spatial[:, 0::2, 1::2, :]
            f3 = spatial[:, 1::2, 1::2, :]
            spatial = torch.cat([f0, f1, f2, f3], -1)  # (B, H/2*W/2, 4*C)
            spatial = spatial.view(batch_size, -1, 4 * num_channels)

            # ── Re-prepend upsampled prompts, then reduction+norm ──
            combined = torch.cat([prompt_emb, spatial], dim=1)  # (B, np + H/2*W/2, 4*C)
            combined = downsample.reduction(combined)  # (B, np + H/2*W/2, 2*C)
            combined = downsample.norm(combined)

            return combined

        return patched_downsample_forward

    # ── Patch Swinv2Stage.forward: inject deep prompts between blocks ──
    def make_patched_stage_forward(stage, stage_idx, deep_prompt_param, initial_prompt_param,
                                   dropout_fn, np_tokens):
        def patched_stage_forward(hidden_states, input_dimensions, output_attentions=False):
            height, width = input_dimensions
            B = hidden_states.shape[0]
            num_blocks = len(stage.blocks)

            for i, block in enumerate(stage.blocks):
                if stage_idx == 0:
                    # First stage: block 0 uses initial prompt, rest use deep_prompts[0]
                    if i == 0:
                        # Initial prompt already prepended by embeddings hook
                        pass
                    else:
                        # Replace prompts with deep prompt (deep_prompts[0] has depths[0]-1 entries)
                        deep_p = dropout_fn(deep_prompt_param[i - 1].unsqueeze(0).expand(B, -1, -1))
                        hidden_states = torch.cat([deep_p, hidden_states[:, np_tokens:, :]], dim=1)
                else:
                    # Stages 1-3: replace prompts at every block
                    deep_p = dropout_fn(deep_prompt_param[i].unsqueeze(0).expand(B, -1, -1))
                    hidden_states = torch.cat([deep_p, hidden_states[:, np_tokens:, :]], dim=1)

                layer_outputs = block(hidden_states, input_dimensions, output_attentions)
                hidden_states = layer_outputs[0]

            hidden_states_before_downsampling = hidden_states
            if stage.downsample is not None:
                height_ds, width_ds = (height + 1) // 2, (width + 1) // 2
                output_dimensions = (height, width, height_ds, width_ds)
                hidden_states = stage.downsample(hidden_states_before_downsampling, input_dimensions)
            else:
                output_dimensions = (height, width, height, width)

            stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)
            if output_attentions:
                stage_outputs += layer_outputs[1:]
            return stage_outputs

        return patched_stage_forward

    # ── Apply all patches ──

    # Patch attention forward for all blocks
    for stage_idx, stage in enumerate(swinv2.encoder.layers):
        for block in stage.blocks:
            block.attention.self.forward = make_patched_attention_forward(
                block.attention.self, num_tokens)
            block.forward = make_patched_block_forward(block, num_tokens)

        # Patch downsample
        if stage.downsample is not None:
            stage.downsample.forward = make_patched_downsample_forward(
                stage.downsample, num_tokens)

        # Patch stage forward
        stage.forward = make_patched_stage_forward(
            stage, stage_idx, deep_prompts[stage_idx], initial_prompt,
            prompt_dropout, num_tokens)

    # ── Prepend initial prompt after embeddings ──
    orig_embed_forward = swinv2.embeddings.forward

    def patched_embeddings_forward(*args, **kwargs):
        result = orig_embed_forward(*args, **kwargs)
        # result is (embedding_output, output_dimensions)
        embedding_output, output_dimensions = result[0], result[1]
        B = embedding_output.shape[0]
        prompt = prompt_dropout(initial_prompt.expand(B, -1, -1))
        embedding_output = torch.cat([prompt, embedding_output], dim=1)
        return (embedding_output,) + result[1:]

    swinv2.embeddings.forward = patched_embeddings_forward

    # ── Strip prompts before final pooling ──
    # The VPT repo INCLUDES prompts in avgpool. We do the same — no stripping needed.
    # The pooler (AdaptiveAvgPool1d) averages over all tokens including prompts.

    return model


# ======================= SSF ==============================================
# Ref: SSF-main/models/swin_transformer.py — init_ssf_scale_shift, ssf_ada
# HF SwinV2 differences from repo:
#   - Separate Q/K/V (not fused QKV) → 3 SSFs of stage_dim each
#   - Post-norm (layernorm_before = post-attn, layernorm_after = post-MLP)
#   - PatchMerging: reduction THEN norm (repo is norm then reduction)
# ---------------------------------------------------------------------------

def init_ssf_scale_shift(dim):
    """Exact init from SSF repo."""
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(scale, mean=1, std=0.02)
    nn.init.normal_(shift, std=0.02)
    return scale, shift


def ssf_ada(x, scale, shift):
    """Exact SSF transform from SSF repo."""
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError(f"SSF shape mismatch: x {x.shape} vs scale {scale.shape}")


class SSFWrapper(nn.Module):
    """Wraps any module, applying SSF (scale+shift) to its output."""
    def __init__(self, original_module, dim):
        super().__init__()
        self.original = original_module
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.scale, mean=1, std=0.02)
        nn.init.normal_(self.shift, std=0.02)

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return getattr(self.original, 'bias', None)

    def forward(self, *args, **kwargs):
        x = self.original(*args, **kwargs)
        if isinstance(x, tuple):
            return (ssf_ada(x[0], self.scale, self.shift),) + x[1:]
        return ssf_ada(x, self.scale, self.shift)


def apply_ssf(model, num_classes, config, task_name=""):
    """Apply SSF: learnable scale & shift after every linear/norm output.
    Placement matches SSF-main/models/swin_transformer.py:
      PatchEmbed: after proj, after norm
      Block: after norm1 output, after QKV (or Q,K,V separately), after attn proj,
             after norm2 output, after MLP fc1, after MLP fc2
      PatchMerging: after norm output (different dim due to HF post-reduction norm)
      Final: after layernorm
    """
    freeze_backbone(model)

    embed_dim = model.config.embed_dim  # 128
    swinv2 = model.swinv2

    # ── PatchEmbed: after proj (embed_dim) + after norm (embed_dim) ──
    swinv2.embeddings.patch_embeddings.projection = SSFWrapper(
        swinv2.embeddings.patch_embeddings.projection, embed_dim)
    swinv2.embeddings.norm = SSFWrapper(swinv2.embeddings.norm, embed_dim)

    for stage_idx, stage in enumerate(swinv2.encoder.layers):
        stage_dim = embed_dim * (2 ** stage_idx)  # 128, 256, 512, 1024
        mlp_hidden = int(model.config.mlp_ratio * stage_dim)

        for block in stage.blocks:
            # ── Attention: after Q, K, V (each stage_dim) ──
            # Repo uses fused QKV SSF(3*dim); HF has separate Q/K/V → 3 × SSF(dim)
            block.attention.self.query = SSFWrapper(block.attention.self.query, stage_dim)
            block.attention.self.key   = SSFWrapper(block.attention.self.key,   stage_dim)
            block.attention.self.value = SSFWrapper(block.attention.self.value, stage_dim)
            # After attn output projection
            block.attention.output.dense = SSFWrapper(block.attention.output.dense, stage_dim)

            # ── Post-attn norm (layernorm_before in HF = post-attn) ──
            block.layernorm_before = SSFWrapper(block.layernorm_before, stage_dim)

            # ── MLP: after fc1 (mlp_hidden), after fc2 (stage_dim) ──
            block.intermediate.dense = SSFWrapper(block.intermediate.dense, mlp_hidden)
            block.output.dense = SSFWrapper(block.output.dense, stage_dim)

            # ── Post-MLP norm (layernorm_after in HF = post-MLP) ──
            block.layernorm_after = SSFWrapper(block.layernorm_after, stage_dim)

        # ── PatchMerging: SSF after norm output ──
        # Repo (pre-norm Swin v1): norm(4*dim) → SSF(4*dim) → reduction
        # HF (post-norm SwinV2): reduction → norm(2*dim) → SSF(2*dim)
        if stage.downsample is not None:
            down_out_dim = embed_dim * (2 ** (stage_idx + 1))  # 2*stage_dim
            stage.downsample.norm = SSFWrapper(stage.downsample.norm, down_out_dim)

    # ── Final layernorm ──
    final_dim = embed_dim * (2 ** (len(model.config.depths) - 1))
    swinv2.layernorm = SSFWrapper(swinv2.layernorm, final_dim)

    return model


# ======================= ADAPTFORMER =======================================
# Ref: AdaptFormer-main/models/adapter.py — Adapter class
# Ref: AdaptFormer-main/models/custom_modules.py — Block with parallel adapter
# Paper: "AdaptMLP applies to Swin without modification" (same MLP structure)
# Adapter is parallel to MLP: adapter(hidden_states) added to MLP output.
# ---------------------------------------------------------------------------
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
    """Apply AdaptFormer: parallel adapter in each FFN block.
    HF SwinV2 post-norm block:
      shortcut = x
      attn_out = attention(x) + window ops
      x = shortcut + drop_path(layernorm_before(attn_out))  # post-norm
      mlp_out = intermediate(x) → output(mlp_out)
      x = x + drop_path(layernorm_after(mlp_out))           # post-norm
    Adapter: parallel to MLP, takes x (after attn residual), adds to final output.
    """
    freeze_backbone(model)

    embed_dim  = model.config.embed_dim
    bottleneck = config["adapter_bottleneck"]
    dropout    = config["adapter_dropout"]
    scalar     = config["adapter_scalar"]

    adapters = nn.ModuleList()
    swinv2 = model.swinv2

    for stage_idx, stage in enumerate(swinv2.encoder.layers):
        stage_dim = embed_dim * (2 ** stage_idx)

        for block_idx, block in enumerate(stage.blocks):
            adapter = Adapter(
                d_model=stage_dim,
                bottleneck=bottleneck,
                dropout=dropout,
                adapter_scalar=scalar,
                adapter_layernorm_option="none",
            )
            adapters.append(adapter)

            # Capture references for closure
            orig_forward = block.forward

            def make_adapter_forward(orig_fwd, adapt, blk):
                def new_forward(hidden_states, input_dimensions, output_attentions=False):
                    """Replicate block forward with parallel adapter on MLP."""
                    height, width = input_dimensions
                    batch_size, _, channels = hidden_states.size()
                    shortcut = hidden_states

                    # ── Window attention (same as original) ──
                    hidden_states = hidden_states.view(batch_size, height, width, channels)
                    hidden_states, pad_values = blk.maybe_pad(hidden_states, height, width)
                    _, height_pad, width_pad, _ = hidden_states.shape

                    if blk.shift_size > 0:
                        shifted = torch.roll(hidden_states, shifts=(-blk.shift_size, -blk.shift_size), dims=(1, 2))
                    else:
                        shifted = hidden_states

                    from transformers.models.swinv2.modeling_swinv2 import window_partition, window_reverse
                    windows = window_partition(shifted, blk.window_size)
                    windows = windows.view(-1, blk.window_size * blk.window_size, channels)
                    attn_mask = blk.get_attn_mask(height_pad, width_pad, dtype=shortcut.dtype)
                    if attn_mask is not None:
                        attn_mask = attn_mask.to(windows.device)

                    attention_outputs = blk.attention(windows, attn_mask, output_attentions)
                    attention_output = attention_outputs[0]

                    attention_output = attention_output.view(-1, blk.window_size, blk.window_size, channels)
                    shifted_back = window_reverse(attention_output, blk.window_size, height_pad, width_pad)

                    if blk.shift_size > 0:
                        attention_windows = torch.roll(shifted_back, shifts=(blk.shift_size, blk.shift_size), dims=(1, 2))
                    else:
                        attention_windows = shifted_back

                    was_padded = pad_values[3] > 0 or pad_values[5] > 0
                    if was_padded:
                        attention_windows = attention_windows[:, :height, :width, :].contiguous()

                    attention_windows = attention_windows.view(batch_size, height * width, channels)

                    # ── Post-norm attention residual ──
                    hidden_states = shortcut + blk.drop_path(blk.layernorm_before(attention_windows))

                    # ── Parallel adapter (from hidden_states, no residual) ──
                    adapt_out = adapt(hidden_states, add_residual=False)

                    # ── MLP + post-norm ──
                    layer_output = blk.intermediate(hidden_states)
                    layer_output = blk.output(layer_output)
                    layer_output = hidden_states + blk.drop_path(blk.layernorm_after(layer_output))

                    # ── Add adapter output ──
                    layer_output = layer_output + adapt_out

                    outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
                    return outputs

                return new_forward

            block.adapter = adapter
            block.forward = make_adapter_forward(orig_forward, adapter, block)

    model.adapters = adapters

    # BN before classifier (AdaptFormer/MAE protocol)
    final_dim = embed_dim * (2 ** (len(model.config.depths) - 1))
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(final_dim, affine=False, eps=1e-6),
        model.classifier,
    )
    return model


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

    if method == "full_finetune":
        model = apply_full_finetune(model, num_classes, config)
    elif method == "linear_probe":
        model = apply_linear_probe(model, num_classes, config)
    elif method == "vpt_deep":
        model = apply_vpt_deep(model, num_classes, config)
    elif method == "ssf":
        model = apply_ssf(model, num_classes, config, task_name)
    elif method == "adaptformer":
        model = apply_adaptformer(model, num_classes, config)
    elif method == "cft":
        model = apply_cft(model, num_classes, config, selected_nodes, nodes_map, task_name)
    else:
        raise ValueError(f"Unknown method: {method}")

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
print("Building test model (linear probe)...")
_m = build_model("linear_probe", 10, CONFIG)
print(f"  Output shape: {_m(torch.randn(1, 3, CONFIG['image_size'], CONFIG['image_size']).to(device)).logits.shape}")
del _m; gc.collect(); torch.cuda.empty_cache()
print("✅ Model builder works.")