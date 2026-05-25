"""
VTAB-1K Fine-Tuning Benchmark — CFT circuit discovery via EAP-IG

Two variants:
  - discover_circuits_eap_ig: uses log-prob difference metric
  - discover_circuits_eap_ig_v1: uses GT logit metric with KL-style normalization
"""
from collections import defaultdict
import time

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# Pre-train classifier head (linear probe) for meaningful discovery gradients
# =============================================================================
def pretrain_classifier(model, dataset, device, num_epochs=5, lr=1e-3, batch_size=256):
    """Train only the classifier head before circuit discovery.

    A randomly initialized classifier produces near-random gradients, making
    EAP-IG attributions unreliable.  A brief linear-probe warmup gives the
    head meaningful weights so that d(metric)/d(activation) points in
    task-relevant directions.
    """
    print(f"  Pre-training classifier head ({num_epochs} epochs, lr={lr}) ...")

    # Freeze backbone, train only classifier
    for name, p in model.named_parameters():
        p.requires_grad = "classifier" in name

    model.train()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr,
    )

    n = len(dataset)
    indices = list(range(n))

    for epoch in range(num_epochs):
        np.random.shuffle(indices)
        total_loss, correct, total = 0.0, 0, 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            images, labels = [], []
            for idx in batch_idx:
                img, lab = dataset[idx]
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                images.append(img)
                labels.append(lab if isinstance(lab, int) else lab.item())

            imgs = torch.cat(images, dim=0).to(device)
            labs = torch.tensor(labels, dtype=torch.long, device=device)

            logits = model(pixel_values=imgs).logits
            loss = F.cross_entropy(logits, labs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labs)
            correct += (logits.argmax(dim=-1) == labs).sum().item()
            total += len(labs)

        acc = 100.0 * correct / total
        avg_loss = total_loss / total
        print(f"    Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.3f} acc={acc:.1f}%")

    # Unfreeze all params (caller is responsible for model.eval())
    for p in model.parameters():
        p.requires_grad = True
    print(f"  Classifier pre-training done.")


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
def compute_logit_difference(logits, labels):
    """Logit(GT) - Logit(NextBest). Returns scalar (mean over batch)."""
    B = logits.shape[0]
    batch_idx = torch.arange(B, device=logits.device)
    gt_logits = logits[batch_idx, labels]

    masked = logits.clone()
    masked[batch_idx, labels] = float("-inf")
    next_best = masked.max(dim=1).values

    return (gt_logits - next_best).mean()


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
def discover_circuits_eap_ig(model, dataset, config, device=None, metric="log_prob_diff",
                             corruption="patch_shuffle", score_norm="param_count",
                             method="eap-ig"):
    """Discover important circuits using EAP-IG (node-level).
    Interpolation at INPUT embedding level only (faithful to Marks et al.).

    metric: "log_prob_diff" (original) | "logit_diff" | "cross_entropy" (loss-based).
    corruption: "patch_shuffle" | "gaussian" | "channel_shuffle" | "multi"
        "multi" averages scores from all three corruption types.
    score_norm: "param_count" (divide by param count) | "rank" (rank-based within type).
    method: "eap-ig" (original EAP-IG with IG path) or
            "eap" (single gradient at clean input, no IG path).
    """
    if device is None:
        device = next(model.parameters()).device

    print(f"\n{'='*70}")
    print(f"CIRCUIT DISCOVERY -- EAP-IG (metric={metric}, corruption={corruption}, "
          f"norm={score_norm}, method={method})")
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
                if metric == "cross_entropy":
                    objective = F.cross_entropy(out.logits, labels_batch)
                elif metric == "log_prob_diff":
                    objective = compute_log_prob_difference(out.logits, labels_batch)
                elif metric == "logit_diff":
                    objective = compute_logit_difference(out.logits, labels_batch)
                else:
                    raise ValueError(
                        f"Unsupported metric '{metric}'. "
                        "Choose from: log_prob_diff, logit_diff, cross_entropy."
                    )
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

    # -- Normalize scores --
    if score_norm == "rank":
        # Rank-based normalization within each type (heads vs MLPs)
        head_scores = sorted(
            [(n, abs(s)) for n, s in node_scores.items() if "head" in n],
            key=lambda x: x[1], reverse=True)
        mlp_scores = sorted(
            [(n, abs(s)) for n, s in node_scores.items() if "mlp" in n],
            key=lambda x: x[1], reverse=True)

        normalized_scores = {}
        for rank_i, (name, _) in enumerate(head_scores):
            normalized_scores[name] = 1.0 - rank_i / max(len(head_scores) - 1, 1)
        for rank_i, (name, _) in enumerate(mlp_scores):
            normalized_scores[name] = 1.0 - rank_i / max(len(mlp_scores) - 1, 1)
    elif score_norm == "mlp_balanced":
        # No param-count division. Discount MLP scores by the multiplicative
        # size ratio (mlp_pc / head_pc) — taken from nodes_map, model-agnostic.
        head_pcs = [v["param_count"] for v in nodes_map.values() if v["type"] == "head"]
        head_pc_baseline = head_pcs[0] if head_pcs else 1
        normalized_scores = {}
        for name, score in node_scores.items():
            info = nodes_map[name]
            if info.get("type") == "mlp":
                ratio = info["param_count"] / head_pc_baseline
                normalized_scores[name] = score / ratio
            else:
                normalized_scores[name] = score
        print(f"  [mlp_balanced] mlp:head param ratio = "
              f"{nodes_map[next(n for n,v in nodes_map.items() if v['type']=='mlp')]['param_count'] / head_pc_baseline:.2f}")
    else:
        # Default: normalize by parameter count
        normalized_scores = {}
        for name, score in node_scores.items():
            pc = nodes_map[name]["param_count"]
            normalized_scores[name] = score / pc if pc > 0 else 0.0

    sorted_nodes = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop 20 nodes by normalized EAP-IG score ({score_norm}):")
    for i, (name, score) in enumerate(sorted_nodes[:20], 1):
        raw = node_scores[name]
        pc = nodes_map[name]["param_count"]
        print(f"  {i:2d}. {name:<25s} norm={score:.4e}  raw={raw:.4e}  params={pc:,}")

    return {
        "sorted_nodes": sorted_nodes,
        "node_scores_raw": node_scores,
        "node_scores_normalized": normalized_scores,
        "nodes_map": nodes_map,
        "method": f"EAP-IG-{metric}",
    }


# =============================================================================
# EAP-IG v1 (GT logit, repo-faithful, rank-based normalization)
# =============================================================================
def discover_circuits_eap_ig_v1(model, dataset, config, device=None):
    """EAP-IG node-level circuit discovery, faithful to the Marks et al. repo.
    Uses KL-divergence metric for better layer distribution of scores.
    """
    if device is None:
        device = next(model.parameters()).device

    print(f"\n{'='*70}")
    print("CIRCUIT DISCOVERY -- EAP-IG (KL-divergence, repo-faithful)")
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
    act_norms = {name: 0.0 for name in nodes_map}
    all_idx = list(range(len(dataset)))
    sample_idx = np.random.choice(all_idx, min(num_samples, len(all_idx)), replace=False)
    total_batches = (len(sample_idx) + batch_size - 1) // batch_size if len(sample_idx) > 0 else 0
    print(f"  Discovery start: {total_batches} batches")
    t_start = time.time()

    total_items = 0

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
        cur_bs = clean_batch.shape[0]
        total_items += cur_bs
        corrupt_batch = create_patch_shuffled_image(clean_batch, patch_size=patch_size)

        act_diff = {}
        embed_storage = {}

        def make_fwd_hook(storage, name, add=True):
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                if add:
                    storage[name] = storage.get(name, 0) + o.detach()
                else:
                    storage[name] = storage.get(name, 0) - o.detach()
            return hook

        def make_embed_capture_hook(storage, key):
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                storage[key] = o.detach()
            return hook

        # Corrupted forward pass: ADD activations
        handles = []
        for i, layer_mod in enumerate(vit.encoder.layer):
            handles.append(layer_mod.attention.output.register_forward_hook(
                make_fwd_hook(act_diff, f"layer_{i}_attn", add=True)))
            handles.append(layer_mod.output.register_forward_hook(
                make_fwd_hook(act_diff, f"layer_{i}_mlp", add=True)))
        handles.append(vit.embeddings.register_forward_hook(
            make_embed_capture_hook(embed_storage, "corrupt")))

        with torch.no_grad():
            model(pixel_values=corrupt_batch)
        for h in handles:
            h.remove()

        # Clean forward pass: SUBTRACT activations
        handles = []
        for i, layer_mod in enumerate(vit.encoder.layer):
            handles.append(layer_mod.attention.output.register_forward_hook(
                make_fwd_hook(act_diff, f"layer_{i}_attn", add=False)))
            handles.append(layer_mod.output.register_forward_hook(
                make_fwd_hook(act_diff, f"layer_{i}_mlp", add=False)))
        handles.append(vit.embeddings.register_forward_hook(
            make_embed_capture_hook(embed_storage, "clean")))

        with torch.no_grad():
            clean_out = model(pixel_values=clean_batch)
            clean_logits = clean_out.logits.detach()
        for h in handles:
            h.remove()

        embed_clean = embed_storage["clean"]
        embed_corrupt = embed_storage["corrupt"]

        for key in act_diff:
            if "_attn" in key:
                layer_i = int(key.split("_")[1])
                for h_idx in range(n_heads):
                    slice_norm = act_diff[key][:, :, h_idx*d_head:(h_idx+1)*d_head].float().norm().item()
                    act_norms[f"layer_{layer_i}_head_{h_idx}"] += slice_norm
            else:
                layer_i = int(key.split("_")[1])
                act_norms[f"layer_{layer_i}_mlp"] += act_diff[key].float().norm().item()

        for step_k in range(1, ig_steps + 1):
            alpha = step_k / ig_steps
            embed_interp = embed_corrupt + alpha * (embed_clean - embed_corrupt)

            def make_embed_interp_hook(interp_val):
                def hook(mod, inp, out):
                    return interp_val.clone().requires_grad_(True) + out * 0
                return hook

            step_scores = {}

            def make_bwd_hook_attn(layer_i):
                def hook(mod, grad_input, grad_output):
                    grad = grad_output[0]
                    if grad is None:
                        return
                    grad = grad.detach()
                    diff = act_diff[f"layer_{layer_i}_attn"]
                    attr_per_hidden = (diff * grad).sum(dim=(0, 1))
                    for h_idx in range(n_heads):
                        s = attr_per_hidden[h_idx * d_head : (h_idx + 1) * d_head].sum().item()
                        key = f"layer_{layer_i}_head_{h_idx}"
                        step_scores[key] = step_scores.get(key, 0.0) + s
                return hook

            def make_bwd_hook_mlp(layer_i):
                def hook(mod, grad_input, grad_output):
                    grad = grad_output[0]
                    if grad is None:
                        return
                    grad = grad.detach()
                    diff = act_diff[f"layer_{layer_i}_mlp"]
                    s = (diff * grad).sum().item()
                    key = f"layer_{layer_i}_mlp"
                    step_scores[key] = step_scores.get(key, 0.0) + s
                return hook

            fwd_handles = []
            bwd_handles = []

            fwd_handles.append(vit.embeddings.register_forward_hook(
                make_embed_interp_hook(embed_interp)))

            for i, layer_mod in enumerate(vit.encoder.layer):
                bwd_handles.append(layer_mod.attention.output.register_full_backward_hook(
                    make_bwd_hook_attn(i)))
                bwd_handles.append(layer_mod.layernorm_after.register_full_backward_hook(
                    make_bwd_hook_mlp(i)))

            model.zero_grad()
            out = model(pixel_values=clean_batch)
            labels_batch = torch.tensor(labels, dtype=torch.long, device=device)
            logit_loss = compute_gt_logit(out.logits, labels_batch)
            logit_loss.backward()

            for name in step_scores:
                node_scores[name] += step_scores[name]

            for h in fwd_handles:
                h.remove()
            for h in bwd_handles:
                h.remove()

        if (total_items // cur_bs) % 5 == 0:
            torch.cuda.empty_cache()

    # Normalize: /total_items /ig_steps
    model.zero_grad()
    torch.cuda.empty_cache()
    for name in node_scores:
        node_scores[name] /= max(total_items, 1)
        node_scores[name] /= max(ig_steps, 1)

    for name in act_norms:
        act_norms[name] /= max(total_items, 1)
    elapsed = time.time() - t_start
    print(f"  Discovery done: {total_items} samples in {elapsed:.1f}s")

    head_scores = sorted(
        [(n, abs(s)) for n, s in node_scores.items() if "head" in n],
        key=lambda x: x[1], reverse=True)
    mlp_scores = sorted(
        [(n, abs(s)) for n, s in node_scores.items() if "mlp" in n],
        key=lambda x: x[1], reverse=True)

    normalized_scores = {}
    for rank_i, (name, _) in enumerate(head_scores):
        normalized_scores[name] = 1.0 - rank_i / max(len(head_scores) - 1, 1)
    for rank_i, (name, _) in enumerate(mlp_scores):
        normalized_scores[name] = 1.0 - rank_i / max(len(mlp_scores) - 1, 1)

    sorted_nodes = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop 20 nodes by |normalized| EAP-IG score:")
    for i, (name, score) in enumerate(sorted_nodes[:20], 1):
        raw = node_scores[name]
        pc = nodes_map[name]["param_count"]
        print(f"  {i:2d}. {name:<25s} |norm|={score:.4e}  raw={raw:+.4e}  params={pc:,}")

    # Layer distribution diagnostic
    layer_score_sum = defaultdict(float)
    for name, score in node_scores.items():
        layer_i = int(name.split("_")[1])
        layer_score_sum[layer_i] += abs(score)
    total_score = sum(layer_score_sum.values()) + 1e-12
    print(f"\n  Layer distribution of |scores|:")
    for layer_i in sorted(layer_score_sum.keys()):
        pct = 100 * layer_score_sum[layer_i] / total_score
        bar = "#" * int(pct / 2)
        print(f"    Layer {layer_i:2d}: {pct:5.1f}% {bar}")

    return {
        "sorted_nodes": sorted_nodes,
        "node_scores_raw": node_scores,
        "node_scores_normalized": normalized_scores,
        "nodes_map": nodes_map,
        "method": "EAP-IG-KL",
    }


# =============================================================================
# Node Selection by Parameter Budget
# =============================================================================
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
