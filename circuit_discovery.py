"""
"""
from collections import defaultdict
import time

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# Corruption methods
# =============================================================================
def create_patch_shuffled_image(image_tensor, patch_size=16):
    """Corrupt images by shuffling patches.
    Breaks global structure while preserving local texture.
    """
    B, C, H, W = image_tensor.shape
    dev = image_tensor.device
    n_h, n_w = H // patch_size, W // patch_size
    n_patches = n_h * n_w

    patches = image_tensor.view(B, C, n_h, patch_size, n_w, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = patches.view(B, n_patches, C, patch_size, patch_size)

    shuffled = torch.zeros_like(patches)
    for b in range(B):
        perm = torch.randperm(n_patches, device=dev)
        shuffled[b] = patches[b, perm]

    shuffled = shuffled.view(B, n_h, n_w, C, patch_size, patch_size)
    shuffled = shuffled.permute(0, 3, 1, 4, 2, 5).contiguous()
    return shuffled.view(B, C, H, W)

def create_gaussian_noise_image(image_tensor, **kwargs):
    """Corrupt images with Gaussian noise matching per-image statistics.
    Destroys all structure (spatial + texture/color).
    """
    mean = image_tensor.mean(dim=(2, 3), keepdim=True)
    std = image_tensor.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    return mean + std * torch.randn_like(image_tensor)

def create_channel_shuffled_image(image_tensor, patch_size=16):
    """Corrupt images by shuffling channels within each patch.
    Breaks color/texture while preserving spatial structure.
    """
    B, C, H, W = image_tensor.shape
    dev = image_tensor.device
    n_h, n_w = H // patch_size, W // patch_size

    out = image_tensor.clone()
    patches = out.view(B, C, n_h, patch_size, n_w, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    # patches: [B, n_h, n_w, C, pH, pW]
    for b in range(B):
        perm = torch.randperm(C, device=dev)
        patches[b] = patches[b, :, :, perm, :, :]
    result = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    return result.view(B, C, H, W)

def create_intensity_invert_image(image_tensor, **kwargs):
    """Negate normalized image. Mammo: white-on-black <-> black-on-white."""
    return -image_tensor

def create_cutout_image(image_tensor, mask_size=64, **kwargs):
    """Mask random rectangle to 0 per image. Targets local-lesion priors."""
    B, _, H, W = image_tensor.shape
    out = image_tensor.clone()
    for b in range(B):
        y = int(torch.randint(0, max(H - mask_size, 1), (1,)).item())
        x = int(torch.randint(0, max(W - mask_size, 1), (1,)).item())
        out[b, :, y:y + mask_size, x:x + mask_size] = 0.0
    return out

CORRUPTION_METHODS = {
    "patch_shuffle":    create_patch_shuffled_image,
    "gaussian":         create_gaussian_noise_image,
    "channel_shuffle":  create_channel_shuffled_image,
    "intensity_invert": create_intensity_invert_image,
    "cutout":           create_cutout_image,
}

# =============================================================================
# Metric functions
# =============================================================================

def compute_log_prob_difference(logits, labels):
    """LogProb(GT) - LogProb(NextBest), after softmax.
    Bounded, better-scaled gradients than raw logit diff.
    """
    B = logits.shape[0]
    batch_idx = torch.arange(B, device=logits.device)

    log_probs = F.log_softmax(logits, dim=-1)
    gt_logprobs = log_probs[batch_idx, labels]

    masked = log_probs.clone()
    masked[batch_idx, labels] = float("-inf")
    next_best = masked.max(dim=1).values

    return (gt_logprobs - next_best).mean()

def compute_gt_logit(logits, labels):
    B = logits.shape[0]
    batch_idx = torch.arange(B, device=logits.device)
    return logits[batch_idx, labels].mean()

# =============================================================================
# Node Map Construction
# =============================================================================
def get_vit_nodes(model):
    """Build node map for HuggingFace ViT.
    Each node is either an attention head or an MLP block.
    """
    vit = model.vit
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    hidden = model.config.hidden_size
    d_head = hidden // n_heads

    nodes = {}
    for i, layer in enumerate(vit.encoder.layer):
        head_params = 4 * d_head * hidden + 4 * d_head
        for h in range(n_heads):
            nodes[f"layer_{i}_head_{h}"] = {
                "type": "head",
                "layer_idx": i,
                "head_idx": h,
                "row_start": h * d_head,
                "row_end": (h + 1) * d_head,
                "param_count": head_params,
            }
        mlp_params = sum(
            p.numel()
            for p in list(layer.intermediate.parameters()) + list(layer.output.dense.parameters())
        )
        nodes[f"layer_{i}_mlp"] = {
            "type": "mlp",
            "layer_idx": i,
            "param_count": mlp_params,
        }

    print(f"  {len(nodes)} nodes ({n_heads * n_layers} heads + {n_layers} MLPs)")
    return nodes

# =============================================================================
# EAP-IG Circuit Discovery (log-prob difference variant)
# =============================================================================
def discover_circuits_eap_ig(model, dataset, config, device=None,
                             corruption="patch_shuffle",
                             method="eap-ig"):
    """Discover important circuits using EAP-IG (node-level).
    Interpolation at INPUT embedding level only (faithful to Marks et al.).

    metric: "log_prob_diff" (original) | "logit_diff" | "cross_entropy" (loss-based).
    corruption: "patch_shuffle" | "gaussian" | "channel_shuffle" | "multi"
        "multi" averages scores from all three corruption types.
    method: "eap-ig" (original EAP-IG with IG path)
    """
    if device is None:
        device = next(model.parameters()).device

    print(f"\n{'='*70}")
    print(f"CIRCUIT DISCOVERY -- EAP-IG (corruption={corruption}, "
          f"method={method})")
    print(f"{'='*70}")

    ig_steps   = config["cft_ig_steps"]
    batch_size = config["cft_batch_size"]
    disc_pct   = config["cft_discovery_pct"]
    patch_size = config["patch_size"]

    num_samples = max(1, int(len(dataset) * disc_pct / 100))
    print(f"  Samples: {num_samples}/{len(dataset)}, IG steps: {ig_steps}, Batch: {batch_size}")

    model.eval()
    vit = model.vit
    n_layers = model.config.num_hidden_layers
    n_heads  = model.config.num_attention_heads
    hidden   = model.config.hidden_size
    d_head   = hidden // n_heads

    nodes_map = get_vit_nodes(model)
    node_scores = {name: 0.0 for name in nodes_map}

    all_idx = list(range(len(dataset)))
    sample_idx = np.random.choice(all_idx, min(num_samples, len(all_idx)), replace=False)
    total_batches = (len(sample_idx) + batch_size - 1) // batch_size if len(sample_idx) > 0 else 0
    print(f"  Discovery start: {total_batches} batches")
    t_start = time.time()

    num_batches = 0

    for batch_start in range(0, len(sample_idx), batch_size):
        bidx = sample_idx[batch_start : batch_start + batch_size]
        images, labels = [], []
        for idx in bidx:
            img, lab = dataset[idx]
            if img.dim() == 3:
                img = img.unsqueeze(0)
            images.append(img)
            labels.append(lab if isinstance(lab, int) else lab.item())

        clean_batch = torch.cat(images, dim=0).to(device)
        labels_batch = torch.tensor(labels, dtype=torch.long, device=device)

        if corruption == "multi":
            corrupt_types = list(CORRUPTION_METHODS.keys())
        elif corruption == "multi_med":
            corrupt_types = ["patch_shuffle", "gaussian", "intensity_invert", "cutout"]
        else:
            corrupt_types = [corruption]

        # Accumulate scores across corruption types for this batch
        batch_scores_accum = {name: 0.0 for name in nodes_map}

        for corrupt_type in corrupt_types:
            corrupt_fn = CORRUPTION_METHODS[corrupt_type]
            corrupt_batch = corrupt_fn(clean_batch, patch_size=patch_size)

            # -- Capture clean & corrupt activations + embeddings --
            clean_acts, corrupt_acts = {}, {}
            clean_embed, corrupt_embed = {}, {}

            def make_capture_hook(storage, name):
                def hook(mod, inp, out):
                    storage[name] = out.detach()
                return hook

            handles = []
            for i, layer in enumerate(vit.encoder.layer):
                handles.append(layer.attention.output.dense.register_forward_hook(
                    make_capture_hook(clean_acts, f"layer_{i}_attn")))
                handles.append(layer.output.dense.register_forward_hook(
                    make_capture_hook(clean_acts, f"layer_{i}_mlp")))
            handles.append(vit.embeddings.register_forward_hook(
                make_capture_hook(clean_embed, "embed")))

            with torch.no_grad():
                model(pixel_values=clean_batch)
            for h in handles:
                h.remove()

            handles = []
            for i, layer in enumerate(vit.encoder.layer):
                handles.append(layer.attention.output.dense.register_forward_hook(
                    make_capture_hook(corrupt_acts, f"layer_{i}_attn")))
                handles.append(layer.output.dense.register_forward_hook(
                    make_capture_hook(corrupt_acts, f"layer_{i}_mlp")))
            handles.append(vit.embeddings.register_forward_hook(
                make_capture_hook(corrupt_embed, "embed")))

            with torch.no_grad():
                model(pixel_values=corrupt_batch)
            for h in handles:
                h.remove()

            act_diff = {}
            for key in clean_acts:
                act_diff[key] = corrupt_acts[key] - clean_acts[key]

            embed_clean = clean_embed["embed"]
            embed_corrupt = corrupt_embed["embed"]

            # -- IG: interpolate input embeddings only --
            batch_scores = {name: 0.0 for name in nodes_map}

            if method == "eap":
                alphas = [1.0]  # clean input only, no IG path
            elif method == "eap-ig":
                alphas = [step_k / ig_steps for step_k in range(1, ig_steps + 1)]
            else:
                raise ValueError(
                    f"Unsupported method '{method}'. "
                    "Choose from: eap-ig, eap."
                )

            for alpha in alphas:
                embed_interp = embed_corrupt + alpha * (embed_clean - embed_corrupt)

                def make_embed_interp_hook(interp_val):
                    def hook(mod, inp, out):
                        return interp_val.clone().requires_grad_(True) + out * 0
                    return hook

                step_score_accum = {name: 0.0 for name in nodes_map}

                def make_bwd_hook(name):
                    def hook(mod, grad_input, grad_output):
                        grad = grad_output[0].detach()
                        diff = act_diff[name]
                        attr = (diff * grad).mean(dim=(0, 1))
                        if name.endswith("_attn"):
                            layer_i = int(name.split("_")[1])
                            for h_idx in range(n_heads):
                                s = attr[h_idx * d_head : (h_idx + 1) * d_head].sum().item()
                                step_score_accum[f"layer_{layer_i}_head_{h_idx}"] += abs(s)
                        else:
                            layer_i = int(name.split("_")[1])
                            step_score_accum[f"layer_{layer_i}_mlp"] += abs(attr.sum().item())
                    return hook

                fwd_handles = []
                bwd_handles = []

                fwd_handles.append(vit.embeddings.register_forward_hook(
                    make_embed_interp_hook(embed_interp)))

                for i, layer in enumerate(vit.encoder.layer):
                    bwd_handles.append(layer.attention.output.dense.register_full_backward_hook(
                        make_bwd_hook(f"layer_{i}_attn")))
                    bwd_handles.append(layer.output.dense.register_full_backward_hook(
                        make_bwd_hook(f"layer_{i}_mlp")))

                model.zero_grad()
                out = model(pixel_values=clean_batch)
                objective = compute_log_prob_difference(out.logits, labels_batch)
                objective.backward()

                for name in batch_scores:
                    batch_scores[name] += step_score_accum[name]

                for h in fwd_handles:
                    h.remove()
                for h in bwd_handles:
                    h.remove()

            for name in batch_scores:
                batch_scores[name] /= len(alphas)

            for name in batch_scores_accum:
                batch_scores_accum[name] += batch_scores[name]

        # Average across corruption types
        for name in node_scores:
            node_scores[name] += batch_scores_accum[name] / len(corrupt_types)
        num_batches += 1

        if num_batches % 5 == 0:
            torch.cuda.empty_cache()

    model.zero_grad()
    torch.cuda.empty_cache()
    for name in node_scores:
        node_scores[name] /= max(num_batches, 1)
    elapsed = time.time() - t_start
    print(f"  Discovery done: {num_batches} batches in {elapsed:.1f}s")

    # -- Normalize scores: divide MLP scores by (mlp_params / head_params) so
    # every node score is per equivalent head-sized chunk of parameters. --
    head_pcs = [v["param_count"] for v in nodes_map.values() if v["type"] == "head"]
    head_pc_baseline = head_pcs[0] if head_pcs else 1
    normalized_scores = {}
    for name, score in node_scores.items():
        info = nodes_map[name]
        if info.get("type") == "mlp":
            normalized_scores[name] = score / (info["param_count"] / head_pc_baseline)
        else:
            normalized_scores[name] = score

    sorted_nodes = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop 20 nodes by normalized EAP-IG score:")
    for i, (name, score) in enumerate(sorted_nodes[:20], 1):
        raw = node_scores[name]
        pc = nodes_map[name]["param_count"]
        print(f"  {i:2d}. {name:<25s} norm={score:.4e}  raw={raw:.4e}  params={pc:,}")

    return {
        "sorted_nodes": sorted_nodes,
        "node_scores_raw": node_scores,
        "node_scores_normalized": normalized_scores,
        "nodes_map": nodes_map,
        "method": "EAP-IG",
    }

def select_nodes_by_param_budget(sorted_nodes, nodes_map, total_params, target_pct):
    """Pick top-1 node per layer first, then fill remaining budget from global ranking."""
    budget = int(total_params * target_pct / 100)
    n_layers = max(nodes_map[n]["layer_idx"] for n in nodes_map) + 1

    # Step 1a: Best head per layer that fits the budget (heads first; cheap)
    selected = set()
    used = 0
    for layer_i in range(n_layers):
        for name, score in sorted_nodes:
            if (nodes_map[name]["layer_idx"] == layer_i and
                    nodes_map[name].get("type") == "head"):
                pc = nodes_map[name]["param_count"]
                if used + pc <= budget:
                    selected.add(name); used += pc
                    break  # added best-fit head for this layer
    # Step 1b: Top-scoring MLP across the whole model (just one)
    for name, score in sorted_nodes:
        if nodes_map[name].get("type") == "mlp":
            pc = nodes_map[name]["param_count"]
            if used + pc <= budget:
                selected.add(name); used += pc
            break  # only the top-scoring MLP globally

    # Step 2: Fill remaining budget from global ranking
    for name, score in sorted_nodes:
        if used >= budget:
            break
        if name in selected:
            continue
        pc = nodes_map[name]["param_count"]
        if used + pc <= budget:
            selected.add(name)
            used += pc

    print(f"  CFT budget: {target_pct}% of {total_params:,} = {budget:,}")
    print(f"  Selected {len(selected)} nodes, {used:,} params ({100*used/total_params:.2f}%)")
    print(f"    MLPs: {sum(1 for n in selected if 'mlp' in n)}")
    print(f"    Heads: {sum(1 for n in selected if 'head' in n)}")

    layer_counts = defaultdict(int)
    for name in selected:
        layer_i = int(name.split("_")[1])
        layer_counts[layer_i] += 1
    print(f"  Selected nodes per layer:")
    for li in range(n_layers):
        cnt = layer_counts.get(li, 0)
        bar = "#" * cnt
        print(f"    Layer {li:2d}: {cnt:2d} nodes {bar}")

    return selected, used
