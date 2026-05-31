"""
Gemma PEFT methods — extracted verbatim from CFT_Gemma3_4B_IT_CUB200
"""
from collections import defaultdict
import torch
import torch.nn as nn

# =============================================================================
# METHOD IMPLEMENTATIONS FOR GEMMA-3 VLM
# =============================================================================

def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False



def apply_cft(model, selected_nodes, nodes_map):
    """Freeze everything, unfreeze only discovered circuit nodes."""
    freeze_all(model)

    lang_model = model.model.language_model
    hidden_size = model.config.text_config.hidden_size
    num_heads = model.config.text_config.num_attention_heads
    num_kv_heads = model.config.text_config.num_key_value_heads
    head_dim = model.config.text_config.head_dim
    kv_group_size = num_heads // num_kv_heads

    model._cft_grad_hooks = []

    # Group by layer
    selected_heads_per_layer = defaultdict(set)
    selected_mlps = set()
    for node_name in selected_nodes:
        info = nodes_map[node_name]
        if info["type"] == "head":
            selected_heads_per_layer[info["layer_idx"]].add(info["head_idx"])
        elif info["type"] == "mlp":
            selected_mlps.add(info["layer_idx"])
    #unfreeze layer nrom its no more than 0.005% of the model
    for name, param in model.named_parameters():
        if "norm" in name.lower():
            param.requires_grad = True

    # Unfreeze attention with gradient masks
    for layer_idx, head_set in selected_heads_per_layer.items():
        layer = lang_model.layers[layer_idx]
        attn = layer.self_attn

        # Q projection: mask for selected heads
        q_mask = torch.zeros(num_heads * head_dim, device="cpu", dtype=torch.bfloat16)
        for h in head_set:
            q_mask[h * head_dim:(h + 1) * head_dim] = 1.0

        attn.q_proj.weight.requires_grad = True
        if attn.q_proj.bias is not None:
            attn.q_proj.bias.requires_grad = True
        model._cft_grad_hooks.append(
            attn.q_proj.weight.register_hook(lambda g, m=q_mask: g * m.to(g.device).unsqueeze(1)))

        # K/V: GQA — map heads to kv_heads
        kv_mask = torch.zeros(num_kv_heads * head_dim, device="cpu", dtype=torch.bfloat16)
        for h in head_set:
            kv_head = h // kv_group_size
            kv_mask[kv_head * head_dim:(kv_head + 1) * head_dim] = 1.0

        for proj in [attn.k_proj, attn.v_proj]:
            proj.weight.requires_grad = True
            if proj.bias is not None:
                proj.bias.requires_grad = True
            model._cft_grad_hooks.append(
                proj.weight.register_hook(lambda g, m=kv_mask: g * m.to(g.device).unsqueeze(1)))

        # O projection — mask columns (head dim)
        attn.o_proj.weight.requires_grad = True
        if attn.o_proj.bias is not None:
            attn.o_proj.bias.requires_grad = True
        o_mask = q_mask.clone()
        model._cft_grad_hooks.append(
            attn.o_proj.weight.register_hook(lambda g, m=o_mask: g * m.to(g.device).unsqueeze(0)))
    # Unfreeze MLPs
    for layer_idx in selected_mlps:
        layer = lang_model.layers[layer_idx]
        for param in layer.mlp.parameters():
            param.requires_grad = True

    # Unfreeze all RMSNorm/LayerNorm (small overhead, helps stability)
    for name, param in model.named_parameters():
        if "norm" in name.lower():
            param.requires_grad = True

    effective_trainable = used_params  # from select_nodes_by_param_budget
    norm_params = sum(p.numel() for n, p in model.named_parameters() if "norm" in n.lower() and p.requires_grad)
    effective_trainable += norm_params
    print(f"  [CFT] Effective trainable: {effective_trainable:,} / {TOTAL_PARAMS:,} ({100*effective_trainable/TOTAL_PARAMS:.2f}%)")
    return model
