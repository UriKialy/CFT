"""
Gemma circuit discovery (EAP-IG) — extracted verbatim from
CFT_Gemma3_4B_IT_CUB200.ipynb cell 13.

Requires the helpers from gemma_utils.py (prompt building, answer matching)
to be in scope when discover_circuits_eap_ig is called.
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import defaultdict

# =============================================================================
# EAP-IG Circuit Discovery for Gemma-3 VLM
# =============================================================================
# Nodes: each attention head + each MLP block in the language model
# Clean: (image_A, prompt, answer_A) — activations push toward answer_A
# CF:    (image_B, prompt, answer_B) — activations push toward answer_B
# where B is most-confused class for A's class
# =============================================================================

def get_gemma_nodes(model):
    """Enumerate all attention heads + MLP blocks in Gemma's language model."""
    nodes = {}
    lang_model = model.model.language_model  # the transformer decoder

    num_layers = len(lang_model.layers)
    hidden_size = model.config.text_config.hidden_size
    num_heads = model.config.text_config.num_attention_heads
    num_kv_heads = model.config.text_config.num_key_value_heads
    head_dim  = model.config.text_config.head_dim

    for layer_idx in range(num_layers):
        layer = lang_model.layers[layer_idx]

        # Attention heads
        for h in range(num_heads):
            node_name = f"layer_{layer_idx}_head_{h}"
            # Param count per head: Q rows + K/V rows (GQA) + O cols
            # GQA: num_kv_heads may be < num_heads; each kv_head serves (num_heads // num_kv_heads) q_heads
            q_params = head_dim * hidden_size + head_dim  # q_proj rows for this head
            kv_group_size = num_heads // num_kv_heads
            kv_share = 1.0 / kv_group_size  # fractional ownership
            k_params = (head_dim * hidden_size + head_dim) * kv_share
            v_params = (head_dim * hidden_size + head_dim) * kv_share
            o_params = hidden_size * head_dim  # o_proj cols for this head (no bias typically)
            head_params = int(q_params + k_params + v_params + o_params)

            nodes[node_name] = {
                "type": "head",
                "layer_idx": layer_idx,
                "head_idx": h,
                "head_dim": head_dim,
                "param_count": head_params,
            }

        # MLP
        mlp_params = sum(p.numel() for p in layer.mlp.parameters())
        node_name = f"layer_{layer_idx}_mlp"
        nodes[node_name] = {
            "type": "mlp",
            "layer_idx": layer_idx,
            "param_count": mlp_params,
        }

    total_heads = sum(1 for n in nodes if "head" in n)
    total_mlps = sum(1 for n in nodes if "mlp" in n)
    print(f"  ✓ {len(nodes)} nodes ({total_heads} heads + {total_mlps} MLPs) across {num_layers} layers")
    return nodes


def build_cf_pairs(dataset, task_name, most_confused_map):
    """Build clean/counterfactual pairs using most-confused class mapping."""
    samples_by_class = dataset.get_samples_by_class()
    pairs = []  # list of (clean_idx, cf_idx)

    for idx in range(len(dataset)):
        clean_label = dataset.get_label(idx)
        cf_class = most_confused_map.get(clean_label, (clean_label + 1) % dataset.num_classes)

        if cf_class in samples_by_class and len(samples_by_class[cf_class]) > 0:
            cf_idx = np.random.choice(samples_by_class[cf_class])
        else:
            # Fallback: any different-class sample
            other_classes = [c for c in samples_by_class if c != clean_label]
            if other_classes:
                cf_class = np.random.choice(other_classes)
                cf_idx = np.random.choice(samples_by_class[cf_class])
            else:
                cf_idx = (idx + 1) % len(dataset)
        pairs.append((idx, cf_idx))

    return pairs


def prepare_vlm_input(image, task_name):
    """Prepare processor input for a single image with Strategy 3 prompt."""
    question = build_prompt_strategy3(task_name)
    messages = build_messages(image, question)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    return {k: v.to(model.device) for k, v in inputs.items()}


def get_target_token_id(task_name, label_idx):
    """Get the token ID for the first token of the expected answer."""
    answer_text = get_answer_token(task_name, label_idx)
    token_ids = processor.tokenizer.encode(answer_text, add_special_tokens=False)
    return token_ids[0] if token_ids else None


def discover_circuits_eap_ig(model, train_dataset, task_name, config, most_confused_map):
    """
    EAP-IG circuit discovery for Gemma VLM.
    Clean: (correct image, prompt) -> correct answer activations
    CF: (confused-class image, prompt) -> confused answer activations
    """
    print(f"\n{'='*70}")
    print(f"CIRCUIT DISCOVERY — EAP-IG for {task_name}")
    print(f"{'='*70}")

    ig_steps = config["cft_ig_steps"]
    disc_pct = config["cft_discovery_pct"]

    num_samples = max(1, int(len(train_dataset) * disc_pct / 100))
    print(f"  Samples: {num_samples}/{len(train_dataset)}, IG steps: {ig_steps}")

    model.eval()
    lang_model = model.model.language_model
    num_layers = len(lang_model.layers)
    hidden_size = model.config.text_config.hidden_size
    num_heads = model.config.text_config.num_attention_heads
    head_dim = model.config.text_config.head_dim

    nodes_map = get_gemma_nodes(model)
    node_scores = {name: 0.0 for name in nodes_map}

    # Build CF pairs
    cf_pairs = build_cf_pairs(train_dataset, task_name, most_confused_map)
    np.random.seed(42)
    sample_indices = np.random.choice(len(cf_pairs), min(num_samples, len(cf_pairs)), replace=False)

    num_processed = 0

    for sample_i in tqdm(sample_indices, desc="  EAP-IG"):
        clean_idx, cf_idx = cf_pairs[sample_i]

        clean_img = train_dataset.get_pil_image(clean_idx)
        clean_label = train_dataset.get_label(clean_idx)
        cf_img = train_dataset.get_pil_image(cf_idx)
        cf_label = train_dataset.get_label(cf_idx)

        # Get target token IDs for logit difference
        clean_token_id = get_target_token_id(task_name, clean_label)
        cf_token_id = get_target_token_id(task_name, cf_label)
        if clean_token_id is None or cf_token_id is None:
            continue

        # Prepare inputs
        clean_inputs = prepare_vlm_input(clean_img, task_name)
        cf_inputs = prepare_vlm_input(cf_img, task_name)

        # ── Capture clean & CF activations ──
        clean_acts, cf_acts = {}, {}

        def make_capture_hook(storage, name):
            def hook(mod, inp, out):
                if isinstance(out, tuple):
                    storage[name] = out[0].detach()
                else:
                    storage[name] = out.detach()
            return hook

        handles = []
        for layer_idx in range(num_layers):
            layer = lang_model.layers[layer_idx]
            handles.append(layer.self_attn.o_proj.register_forward_hook(
                make_capture_hook(clean_acts, f"layer_{layer_idx}_attn")))
            handles.append(layer.mlp.register_forward_hook(
                make_capture_hook(clean_acts, f"layer_{layer_idx}_mlp")))

        with torch.no_grad():
            clean_out = model(**clean_inputs)
        for h in handles:
            h.remove()

        handles = []
        for layer_idx in range(num_layers):
            layer = lang_model.layers[layer_idx]
            handles.append(layer.self_attn.o_proj.register_forward_hook(
                make_capture_hook(cf_acts, f"layer_{layer_idx}_attn")))
            handles.append(layer.mlp.register_forward_hook(
                make_capture_hook(cf_acts, f"layer_{layer_idx}_mlp")))

        with torch.no_grad():
            cf_out = model(**cf_inputs)
        for h in handles:
            h.remove()

        # ── IG: interpolate and accumulate ──
        batch_scores = {name: 0.0 for name in nodes_map}

        for step_k in range(1, ig_steps + 1):
            alpha = step_k / ig_steps
            step_grads = {}

            def make_interp_hook(name, alpha_val):
                def hook(mod, inp, out):
                    if name not in clean_acts or name not in cf_acts:
                        return out
                    # Match shapes (sequences may differ slightly)
                    c_act = clean_acts[name]
                    cf_act = cf_acts[name]
                    min_len = min(c_act.shape[1], cf_act.shape[1])
                    interp = alpha_val * c_act[:, :min_len] + (1 - alpha_val) * cf_act[:, :min_len]

                    if isinstance(out, tuple):
                        orig = out[0]
                    else:
                        orig = out
                    # Pad/trim to match original
                    result = orig.clone()
                    result[:, :min_len] = interp
                    result.requires_grad_(True)
                    step_grads[name] = result
                    if isinstance(out, tuple):
                        return (result,) + out[1:]
                    return result
                return hook

            def make_grad_hook(name):
                def hook(grad):
                    step_grads[name + '_grad'] = grad.detach()
                return hook

            fwd_handles, grad_handles = [], []
            for layer_idx in range(num_layers):
                layer = lang_model.layers[layer_idx]
                fwd_handles.append(layer.self_attn.o_proj.register_forward_hook(
                    make_interp_hook(f"layer_{layer_idx}_attn", alpha)))
                fwd_handles.append(layer.mlp.register_forward_hook(
                    make_interp_hook(f"layer_{layer_idx}_mlp", alpha)))

            model.zero_grad()
            out = model(**clean_inputs)

            # Register grad hooks
            for name, tensor in step_grads.items():
                if not name.endswith('_grad') and tensor.requires_grad:
                    grad_handles.append(tensor.register_hook(make_grad_hook(name)))

            # # Logit difference: logit(clean_answer) - logit(cf_answer)
            # last_logits = out.logits[0, -1, :]  # logits at last position
            # logit_diff = last_logits[clean_token_id] - last_logits[cf_token_id]
            # logit_diff.backward()

            # Cross-entropy loss on the clean answer token
            last_logits = out.logits[0, -1, :].unsqueeze(0)  # (1, vocab_size)
            target = torch.tensor([clean_token_id], device=last_logits.device)
            loss = F.cross_entropy(last_logits, target)
            loss.backward()

            # Accumulate scores
            for layer_idx in range(num_layers):
                for prefix in ['attn', 'mlp']:
                    key = f"layer_{layer_idx}_{prefix}"
                    grad_key = key + '_grad'
                    if grad_key in step_grads and key in clean_acts and key in cf_acts:
                        c_act = clean_acts[key]
                        cf_act = cf_acts[key]
                        min_len = min(c_act.shape[1], cf_act.shape[1])
                        diff = c_act[:, :min_len] - cf_act[:, :min_len]
                        grad = step_grads[grad_key][:, :min_len]
                        attr = (diff * grad).mean(dim=(0, 1))  # (hidden_size,)

                        if prefix == 'attn':
                            for h_idx in range(num_heads):
                                s = attr[h_idx * head_dim:(h_idx + 1) * head_dim].sum().item()
                                batch_scores[f"layer_{layer_idx}_head_{h_idx}"] += abs(s)
                        else:
                            batch_scores[f"layer_{layer_idx}_mlp"] += abs(attr.sum().item())

            for h in fwd_handles:
                h.remove()
            for h in grad_handles:
                h.remove()
            step_grads.clear()

        for name in node_scores:
            node_scores[name] += batch_scores[name] / ig_steps
        num_processed += 1

        # Clear VRAM
        del clean_acts, cf_acts, clean_inputs, cf_inputs, clean_out, cf_out
        gc.collect()
        torch.cuda.empty_cache()

    # Normalize by number of samples
    if num_processed > 0:
        for name in node_scores:
            node_scores[name] /= num_processed

    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 20 nodes by EAP-IG score:")
    for i, (name, score) in enumerate(sorted_nodes[:20], 1):
        pc = nodes_map[name]["param_count"]
        ntype = "MLP" if "mlp" in name else "Head"
        print(f"  {i:2d}. {name:<30s} score={score:.4e}  params={pc:,}  ({ntype})")

    return {
        "sorted_nodes": sorted_nodes,
        "node_scores_raw": node_scores,
        "nodes_map": nodes_map,
    }


def select_nodes_by_param_budget(sorted_nodes, nodes_map, total_params, budget_pct):
    """Pick top nodes until cumulative params reach budget_pct of total_params.
    Strategy: first pick the best head from each layer, then fill remaining budget with top nodes.
    """
    budget = int(total_params * budget_pct / 100)

    # --- Phase 1: Best head from each layer ---
    best_head_per_layer = {}
    for name, score in sorted_nodes:
        if "head" not in name:
            continue
        layer_idx = int(name.split("_")[1])
        if layer_idx not in best_head_per_layer or score > best_head_per_layer[layer_idx][1]:
            best_head_per_layer[layer_idx] = (name, score)

    # Sort by score descending so we prioritize the most important layers
    sorted_layers = sorted(best_head_per_layer.items(), key=lambda x: x[1][1], reverse=True)

    selected = set()
    used = 0
    for layer_idx, (name, score) in sorted_layers:
        pc = nodes_map[name]["param_count"]
        if used + pc > budget and used > 0:
            continue
        selected.add(name)
        used += pc

    # --- Phase 2: Fill remaining budget with top-scoring nodes ---
    for name, score in sorted_nodes:
        if name in selected:
            continue
        pc = nodes_map[name]["param_count"]
        if used + pc > budget:
            continue
        selected.add(name)
        used += pc
        if used >= budget:
            break

    n_h = sum(1 for n in selected if "head" in n)
    n_m = sum(1 for n in selected if "mlp" in n)
    print(f"  CFT budget: {budget_pct}% of {total_params:,} = {budget:,}")
    print(f"  Selected {len(selected)} nodes, {used:,} params ({100*used/total_params:.2f}%)")
    print(f"  → {n_h} heads + {n_m} MLPs")
    return selected, used

print("✅ CFT functions defined.")