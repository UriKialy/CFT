"""
Swin circuit discovery (EAP-IG and EAP) — extracted verbatim from
Swin_vtab1k_CFT.ipynb cells 8 and 9.
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# =============================================================================
# CELL 6: CFT — Corruption, EAP-IG Circuit Discovery, Node Selection
# =============================================================================
# Logic adapted from user's DINOv2 implementation for HuggingFace ViT.
# EAP-IG: Edge Attribution Patching with Integrated Gradients (Marks et al.)
# =============================================================================
import torch.nn.functional as F
# ── Corruption: Patch Shuffle ──────────────────────────────────────────────
def create_patch_shuffled_image(image_tensor, patch_size=16):
    """
    Corrupt images by shuffling patches.
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

# ── Logit Difference Metric ───────────────────────────────────────────────
def compute_logit_difference(logits, labels):
    """
    Logit(GT) - Logit(NextBest).
    Returns scalar (mean over batch) for backward.
    """
    B = logits.shape[0]
    batch_idx = torch.arange(B, device=logits.device)
    gt_logits = logits[batch_idx, labels]

    masked = logits.clone()
    masked[batch_idx, labels] = float("-inf")
    next_best = masked.max(dim=1).values

    return (gt_logits - next_best).mean()

def compute_log_prob_difference(logits, labels):
    """
    LogProb(GT) - LogProb(NextBest), after softmax.
    Bounded, better-scaled gradients than raw logit diff.
    """
    B = logits.shape[0]
    batch_idx = torch.arange(B, device=logits.device)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    gt_logprobs = log_probs[batch_idx, labels]

    masked = log_probs.clone()
    masked[batch_idx, labels] = float("-inf")
    next_best = masked.max(dim=1).values

    return (gt_logprobs - next_best).mean()

# ── Node Map Construction ──────────────────────────────────────────────────
def get_swinv2_nodes(model):

    # Each head is one node. row_start/row_end identify this head's
    # rows within the fused QKV weight matrix, used by CFT to
    # selectively unfreeze individual head parameters.

    swinv2 = model.swinv2
    depths = model.config.depths          # e.g. [2, 2, 18, 2]
    num_heads = model.config.num_heads    # e.g. [4, 8, 16, 32]
    embed_dim = model.config.embed_dim    # e.g. 128

    nodes = {}
    total_heads = 0
    total_mlps = 0

    for stage_idx, stage in enumerate(swinv2.encoder.layers):
        stage_dim = embed_dim * (2 ** stage_idx)       # 128, 256, 512, 1024
        n_heads = num_heads[stage_idx]
        d_head = stage_dim // n_heads

        for block_idx, block in enumerate(stage.blocks):
            # ── Attention heads ──
            # Each head slice: Q/K/V rows [h*d_h:(h+1)*d_h] + output.dense cols
            # param_count per head ≈ 4 * d_head * stage_dim + 4 * d_head
            head_params = 4 * d_head * stage_dim + 4 * d_head

            for h in range(n_heads):
                node_name = f"stage_{stage_idx}_block_{block_idx}_head_{h}"
                nodes[node_name] = {
                    "type": "head",
                    "stage_idx": stage_idx,
                    "block_idx": block_idx,
                    "head_idx": h,
                    "row_start": h * d_head,
                    "row_end": (h + 1) * d_head,
                    "stage_dim": stage_dim,
                    "param_count": head_params,
                }
                total_heads += 1

            # ── MLP block ──
            mlp_params = sum(
                p.numel()
                for p in list(block.intermediate.parameters())
                           + list(block.output.dense.parameters())
            )
            node_name = f"stage_{stage_idx}_block_{block_idx}_mlp"
            nodes[node_name] = {
                "type": "mlp",
                "stage_idx": stage_idx,
                "block_idx": block_idx,
                "stage_dim": stage_dim,
                "param_count": mlp_params,
            }
            total_mlps += 1

    n_blocks = sum(depths)
    print(f"  ✓ {len(nodes)} nodes ({total_heads} heads + {total_mlps} MLPs) "
          f"across {len(depths)} stages, {n_blocks} blocks")
    return nodes


# ── EAP-IG Circuit Discovery ──────────────────────────────────────────────
def discover_circuits_eap_ig(model, dataset, config):
    """
    Discover important circuits using EAP-IG (node-level).
    Interpolation at INPUT embedding level only (faithful to Marks et al.).
    """
    print(f"\n{'='*70}")
    print("CIRCUIT DISCOVERY — EAP-IG (Integrated Gradients)")
    print(f"{'='*70}")

    ig_steps   = config["cft_ig_steps"]
    batch_size = config["cft_batch_size"]
    disc_pct   = config["cft_discovery_pct"]
    patch_size = config["patch_size"]

    num_samples = max(1, int(len(dataset) * disc_pct / 100))
    print(f"  Samples: {num_samples}/{len(dataset)}, IG steps: {ig_steps}, Batch: {batch_size}")

    model.eval()
    swinv2 = model.swinv2
    depths = model.config.depths            # [2, 2, 18, 2]
    num_heads = model.config.num_heads      # [4, 8, 16, 32]
    embed_dim = model.config.embed_dim      # 128

    nodes_map = get_swinv2_nodes(model)

    nodes_map = get_swinv2_nodes(model)
    node_scores = {name: 0.0 for name in nodes_map}

    all_idx = list(range(len(dataset)))
    np.random.seed(69)
    torch.manual_seed(69)
    torch.cuda.manual_seed(69)
    sample_idx = np.random.choice(all_idx, min(num_samples, len(all_idx)), replace=False)

    num_batches = 0

    for batch_start in tqdm(range(0, len(sample_idx), batch_size), desc="EAP-IG"):
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
# Rotate corruption type per batch
        corruption_types = [
            create_patch_shuffled_image,
            create_gaussian_noise_image,
            create_channel_shuffled_image,
        ]
        corrupt_fn = corruption_types[num_batches % 3]
        corrupted_images = corrupt_fn(clean_batch, patch_size=config["patch_size"])
        # ── Step 1: Capture clean & corrupt activations + embeddings ──
        clean_acts, corrupt_acts = {}, {}
        clean_embed, corrupt_embed = {}, {}

        def make_capture_hook(storage, name):
            def hook(mod, inp, out):
                if isinstance(out, tuple):
                    storage[name] = out[0].detach()
                else:
                    storage[name] = out.detach()
            return hook

        handles = []
        for stage_idx, stage in enumerate(swinv2.encoder.layers):
            for block_idx, block in enumerate(stage.blocks):
                handles.append(block.attention.output.dense.register_forward_hook(
                    make_capture_hook(clean_acts, f"stage_{stage_idx}_block_{block_idx}_attn")))
                handles.append(block.output.dense.register_forward_hook(
                    make_capture_hook(clean_acts, f"stage_{stage_idx}_block_{block_idx}_mlp")))
        handles.append(swinv2.embeddings.register_forward_hook(
            make_capture_hook(clean_embed, "embed")))

        with torch.no_grad():
            model(pixel_values=clean_batch)
        for h in handles:
            h.remove()

        handles = []
        for stage_idx, stage in enumerate(swinv2.encoder.layers):
            for block_idx, block in enumerate(stage.blocks):
                handles.append(block.attention.output.dense.register_forward_hook(
                    make_capture_hook(corrupt_acts, f"stage_{stage_idx}_block_{block_idx}_attn")))
                handles.append(block.output.dense.register_forward_hook(
                    make_capture_hook(corrupt_acts, f"stage_{stage_idx}_block_{block_idx}_mlp")))
        handles.append(swinv2.embeddings.register_forward_hook(
            make_capture_hook(corrupt_embed, "embed")))

        with torch.no_grad():
            model(pixel_values=corrupted_images)
        for h in handles:
            h.remove()

        # Precompute activation differences: corrupt - clean
        act_diff = {}
        for key in clean_acts:
            act_diff[key] = corrupt_acts[key] - clean_acts[key]

        embed_clean = clean_embed["embed"]
        embed_corrupt = corrupt_embed["embed"]

# ── Step 2: IG — interpolate activations at each layer ──
        batch_scores = {name: 0.0 for name in nodes_map}

        for step_k in range(1, ig_steps + 1):
            alpha = step_k / ig_steps

            step_grads = {}

            def make_interp_fwd_hook(name, alpha_val):
                def hook(mod, inp, out):
                    interp = alpha_val * clean_acts[name] + (1 - alpha_val) * corrupt_acts[name]
                    interp.requires_grad_(True)
                    step_grads[name] = interp
                    return interp
                return hook

            def make_grad_hook(name):
                def hook(grad):
                    step_grads[name + '_grad'] = grad.detach()
                return hook

            fwd_handles = []
            grad_handles = []

            for stage_idx, stage in enumerate(swinv2.encoder.layers):
                for block_idx, block in enumerate(stage.blocks):
                    fwd_handles.append(block.attention.output.dense.register_forward_hook(
                        make_interp_fwd_hook(f"stage_{stage_idx}_block_{block_idx}_attn", alpha)))
                    fwd_handles.append(block.output.dense.register_forward_hook(
                        make_interp_fwd_hook(f"stage_{stage_idx}_block_{block_idx}_mlp", alpha)))

            model.zero_grad()
            out = model(pixel_values=clean_batch)

            # Register grad hooks on interpolated tensors
            for name, tensor in step_grads.items():
                if not name.endswith('_grad') and tensor.requires_grad:
                    grad_handles.append(tensor.register_hook(make_grad_hook(name)))

            # logit_diff = compute_log_prob_difference(out.logits, labels_batch)
            # logit_diff.backward()
            loss = F.cross_entropy(out.logits, labels_batch)
            loss.backward()

            # Accumulate: (clean - corrupt) * grad
            for stage_idx in range(len(depths)):
                stage_dim = embed_dim * (2 ** stage_idx)
                n_heads_stage = num_heads[stage_idx]
                d_head = stage_dim // n_heads_stage
                for block_idx in range(depths[stage_idx]):
                    for prefix in ['attn', 'mlp']:
                        key = f"stage_{stage_idx}_block_{block_idx}_{prefix}"
                        grad_key = key + '_grad'
                        if grad_key in step_grads:
                            diff = clean_acts[key] - corrupt_acts[key]
                            grad = step_grads[grad_key]
                            attr = (diff * grad).mean(dim=(0, 1))

                            if prefix == 'attn':
                                for h_idx in range(n_heads_stage):
                                    s = attr[h_idx * d_head:(h_idx + 1) * d_head].sum().item()
                                    batch_scores[f"stage_{stage_idx}_block_{block_idx}_head_{h_idx}"] += abs(s)
                            else:
                                batch_scores[f"stage_{stage_idx}_block_{block_idx}_mlp"] += abs(attr.sum().item())
            for h in fwd_handles:
                h.remove()
            for h in grad_handles:
                h.remove()
            step_grads.clear()

        for name in node_scores:
            node_scores[name] += batch_scores[name] / ig_steps
        num_batches += 1

# # ── Normalize: scale MLP scores to head-comparable range ──
#     # MLPs have  more params than heads — divide MLP scores by that ratio
#     head_params = nodes_map["layer_0_head_0"]["param_count"]
#     mlp_params = nodes_map["layer_0_mlp"]["param_count"]
#     mlp_ratio = mlp_params / head_params  #

#     normalized_scores = {}
#     for name, score in node_scores.items():
#         if "mlp" in name:
#             normalized_scores[name] = score / mlp_ratio
#         else:
#             normalized_scores[name] = score

#     sorted_nodes = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
    # print(f"\nTop 20 nodes by normalized EAP-IG score:")
    # for i, (name, score) in enumerate(sorted_nodes[:20], 1):
    #     raw = node_scores[name]
    #     pc = nodes_map[name]["param_count"]
    #     print(f"  {i:2d}. {name:<25s} norm={score:.4e}  raw={raw:.4e}  params={pc:,}")

    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 20 nodes by raw EAP-IG score:")
    for i, (name, score) in enumerate(sorted_nodes[:20], 1):
        pc = nodes_map[name]["param_count"]
        ntype = "MLP" if "mlp" in name else "Head"
        print(f"  {i:2d}. {name:<25s} score={score:.4e}  params={pc:,}  ({ntype})")

    return {
        "sorted_nodes": sorted_nodes,
        "node_scores_raw": node_scores,
        "nodes_map": nodes_map,
        "method": "EAP-IG",
    }



def select_nodes_by_param_budget(sorted_nodes, nodes_map, total_params, target_pct,task_name=""):
    """Pick top nodes until cumulative params reach target_pct of total_params.
    Strategy: first pick the best head from each layer, then fill remaining budget with top nodes.
    """
    budget_pct = CFT_TASK_BUDGETS.get(task_name, CONFIG["cft_param_budget"])
    budget = int(budget_pct / 100 * total_params) if budget_pct > 1 else int(budget_pct * total_params)
    best_head_per_block = {}  # (stage_idx, block_idx) -> (name, score)
    for name, score in sorted_nodes:
        if "head" not in name:
            continue
        info = nodes_map[name]
        block_key = (info["stage_idx"], info["block_idx"])
        if block_key not in best_head_per_block or score > best_head_per_block[block_key][1]:
            best_head_per_block[block_key] = (name, score)

    # Sort blocks by their best head's score (descending) so we fill budget fairly
    sorted_blocks = sorted(best_head_per_block.items(), key=lambda x: x[1][1], reverse=True)

    selected = set()
    used = 0
    for block_key, (name, score) in sorted_blocks:
        if name not in nodes_map:
            continue
        pc = nodes_map[name]["param_count"]
        if used + pc > budget and used > 0:
            continue  # skip this block's head if it doesn't fit, try next
        selected.add(name)
        used += pc


    # --- Phase 2: Fill remaining budget with top-scoring nodes (heads or MLPs) ---
    # If a node doesn't fit, skip it and try smaller ones
    for name, score in sorted_nodes:
        if name in selected:
            continue
        if name not in nodes_map:
            continue
        pc = nodes_map[name]["param_count"]
        if used + pc > budget:
            continue  # skip if too big, keep trying smaller nodes
        selected.add(name)
        used += pc
        if used >= budget:
            break

    n_h = sum(1 for n in selected if "head" in n)
    n_m = sum(1 for n in selected if "mlp" in n)
    print(f"  CFT budget: {target_pct}% of {total_params:,} = {budget:,}")
    print(f"  Selected {len(selected)} nodes, {used:,} params ({100*used/total_params:.2f}%)")
    print(f"  → {n_h} heads + {n_m} MLPs")
    return selected, used
print("✅ CFT functions defined.")

def discover_circuits_eap(model, dataset, config):
    """
    Discover important circuits using EAP (Edge Attribution Patching).
    Single forward pass with corrupt input — no interpolation steps.
    Faster and potentially less overfitting than EAP-IG.
    """
    print(f"\n{'='*70}")
    print("CIRCUIT DISCOVERY — EAP (Edge Attribution Patching)")
    print(f"{'='*70}")

    batch_size = config["cft_batch_size"]
    disc_pct   = config["cft_discovery_pct"]
    patch_size = config["patch_size"]

    num_samples = max(1, int(len(dataset) * disc_pct / 100))
    print(f"  Samples: {num_samples}/{len(dataset)}, Batch: {batch_size}")

    model.eval()
    swinv2 = model.swinv2
    depths = model.config.depths
    num_heads = model.config.num_heads
    embed_dim = model.config.embed_dim

    nodes_map = get_swinv2_nodes(model)
    node_scores = {name: 0.0 for name in nodes_map}

    all_idx = list(range(len(dataset)))
    np.random.seed(69)
    torch.manual_seed(69)
    torch.cuda.manual_seed(69)
    sample_idx = np.random.choice(all_idx, min(num_samples, len(all_idx)), replace=False)

    num_batches = 0

    corruption_types = [
        create_patch_shuffled_image,
        create_gaussian_noise_image,
        create_channel_shuffled_image,
    ]

    for batch_start in tqdm(range(0, len(sample_idx), batch_size), desc="EAP"):
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

        corrupt_fn = corruption_types[num_batches % 3]
        corrupted_images = corrupt_fn(clean_batch, patch_size=patch_size)

        # ── Step 1: Clean forward to capture activations ──
        clean_acts = {}

        def make_capture_hook(storage, name):
            def hook(mod, inp, out):
                if isinstance(out, tuple):
                    storage[name] = out[0].detach()
                else:
                    storage[name] = out.detach()
            return hook

        handles = []
        for stage_idx, stage in enumerate(swinv2.encoder.layers):
            for block_idx, block in enumerate(stage.blocks):
                handles.append(block.attention.output.dense.register_forward_hook(
                    make_capture_hook(clean_acts, f"stage_{stage_idx}_block_{block_idx}_attn")))
                handles.append(block.output.dense.register_forward_hook(
                    make_capture_hook(clean_acts, f"stage_{stage_idx}_block_{block_idx}_mlp")))

        with torch.no_grad():
            model(pixel_values=clean_batch)
        for h in handles:
            h.remove()

        # ── Step 2: Corrupt forward to capture activations ──
        corrupt_acts = {}
        handles = []
        for stage_idx, stage in enumerate(swinv2.encoder.layers):
            for block_idx, block in enumerate(stage.blocks):
                handles.append(block.attention.output.dense.register_forward_hook(
                    make_capture_hook(corrupt_acts, f"stage_{stage_idx}_block_{block_idx}_attn")))
                handles.append(block.output.dense.register_forward_hook(
                    make_capture_hook(corrupt_acts, f"stage_{stage_idx}_block_{block_idx}_mlp")))

        with torch.no_grad():
            model(pixel_values=corrupted_images)
        for h in handles:
            h.remove()

        # ── Step 3: Single corrupt forward WITH gradients ──
        # Hook in corrupt activations as requires_grad tensors to get gradients
        step_grads = {}

        def make_replace_hook(name):
            def hook(mod, inp, out):
                act = corrupt_acts[name].clone().requires_grad_(True)
                step_grads[name] = act
                return act
            return hook

        def make_grad_hook(name):
            def hook(grad):
                step_grads[name + '_grad'] = grad.detach()
            return hook

        fwd_handles = []
        grad_handles = []

        for stage_idx, stage in enumerate(swinv2.encoder.layers):
            for block_idx, block in enumerate(stage.blocks):
                fwd_handles.append(block.attention.output.dense.register_forward_hook(
                    make_replace_hook(f"stage_{stage_idx}_block_{block_idx}_attn")))
                fwd_handles.append(block.output.dense.register_forward_hook(
                    make_replace_hook(f"stage_{stage_idx}_block_{block_idx}_mlp")))

        model.zero_grad()
        out = model(pixel_values=corrupted_images)

        for name, tensor in step_grads.items():
            if not name.endswith('_grad') and tensor.requires_grad:
                grad_handles.append(tensor.register_hook(make_grad_hook(name)))

        loss = F.cross_entropy(out.logits, labels_batch)
        loss.backward()

        # ── Step 4: Score = (clean - corrupt) * grad_at_corrupt ──
        for stage_idx in range(len(depths)):
            stage_dim = embed_dim * (2 ** stage_idx)
            n_heads_stage = num_heads[stage_idx]
            d_head = stage_dim // n_heads_stage
            for block_idx in range(depths[stage_idx]):
                for prefix in ['attn', 'mlp']:
                    key = f"stage_{stage_idx}_block_{block_idx}_{prefix}"
                    grad_key = key + '_grad'
                    if grad_key in step_grads:
                        diff = clean_acts[key] - corrupt_acts[key]
                        grad = step_grads[grad_key]
                        attr = (diff * grad).mean(dim=(0, 1))

                        if prefix == 'attn':
                            for h_idx in range(n_heads_stage):
                                s = attr[h_idx * d_head:(h_idx + 1) * d_head].sum().item()
                                node_scores[f"stage_{stage_idx}_block_{block_idx}_head_{h_idx}"] += abs(s)
                        else:
                            node_scores[f"stage_{stage_idx}_block_{block_idx}_mlp"] += abs(attr.sum().item())

        for h in fwd_handles:
            h.remove()
        for h in grad_handles:
            h.remove()
        step_grads.clear()
        num_batches += 1

    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 20 nodes by EAP score:")
    for i, (name, score) in enumerate(sorted_nodes[:20], 1):
        pc = nodes_map[name]["param_count"]
        ntype = "MLP" if "mlp" in name else "Head"
        print(f"  {i:2d}. {name:<25s} score={score:.4e}  params={pc:,}  ({ntype})")

    return {
        "sorted_nodes": sorted_nodes,
        "node_scores_raw": node_scores,
        "nodes_map": nodes_map,
        "method": "EAP",
    }

print("✅ EAP circuit discovery function defined.")