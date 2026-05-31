"""
VTAB-1K Fine-Tuning Benchmark — Training, evaluation, and measurement utilities
"""
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from dataset import GPUCachedDataset
from Utils import count_trainable_params, count_total_params


# =============================================================================
# Training & Evaluation
# =============================================================================
def train_and_evaluate(model, train_ds, test_ds, config, method_name="",
                       test_every=1, task_name="", device=None,
                       ssf_task_configs=None, cft_task_configs=None,
                       use_ddp=False, rank=0, world_size=1,
                       stop_after_epoch=None):
    """Train model and evaluate on test set.
    Returns: dict with accuracy, training_time, inference_time, peak_memory, etc.
    """
    if device is None:
        device = next(model.parameters()).device

    batch_size = config["batch_size"]
    epochs = config["num_epochs"]

    is_cached = isinstance(train_ds, GPUCachedDataset)
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=0 if is_cached else config["num_workers"],
        pin_memory=not is_cached,
    )
    train_sampler = None
    test_sampler = None
    if use_ddp and world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        test_sampler = DistributedSampler(
            test_ds, num_replicas=world_size, rank=rank, shuffle=False
        )

    train_loader = DataLoader(
        train_ds,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        sampler=test_sampler,
        **loader_kwargs,
    )

    # -- Per-method optimizer config (matching original papers) --
    lr_scale = batch_size / 256
    use_smoothing = False

    if method_name == "full_finetune":
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.001, weight_decay=0.0001)
        warmup_epochs = 0

    elif method_name == "linear_probe":
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=5.0 * lr_scale, momentum=0.9, weight_decay=0.0001)
        warmup_epochs = 0

    elif method_name == "vpt_deep":
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=1.0 * lr_scale, momentum=0.9, weight_decay=0.0001)
        warmup_epochs = 10

    elif method_name == "adaptformer":
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.1 * lr_scale, momentum=0.9, weight_decay=0.0)
        warmup_epochs = 20

    elif method_name == "ssf":
        if ssf_task_configs is None:
            ssf_task_configs = {}
        task_cfg = ssf_task_configs.get(task_name, {"lr": 5e-3, "wd": 5e-5})
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=task_cfg["lr"], weight_decay=task_cfg["wd"])
        warmup_epochs = 10

    else:  # cft or default
        if cft_task_configs is None:
            cft_task_configs = {}
        task_cfg = cft_task_configs.get(task_name, {"lr": config["learning_rate"], "wd": config["weight_decay"]})
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        base_model = model.module if hasattr(model, "module") else model
        masked_params = getattr(base_model, "_cft_no_weight_decay_params", None)
        if method_name == "cft" and masked_params:
            masked_ids = {id(p) for p in masked_params if p.requires_grad}
            decay_params = [p for p in trainable_params if id(p) not in masked_ids]
            no_decay_params = [p for p in trainable_params if id(p) in masked_ids]
            optimizer_groups = []
            if decay_params:
                optimizer_groups.append({"params": decay_params, "weight_decay": task_cfg["wd"]})
            if no_decay_params:
                optimizer_groups.append({"params": no_decay_params, "weight_decay": 0.0})
            optimizer = torch.optim.AdamW(optimizer_groups, lr=task_cfg["lr"])
        else:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=task_cfg["lr"], weight_decay=task_cfg["wd"])
        warmup_epochs = 10
        use_smoothing = True

    # -- Linear warmup + cosine decay --
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if use_smoothing:
        ls_val = float(task_cfg.get("label_smoothing", 0.1))
    else:
        ls_val = 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=ls_val)

    # -- Reset peak memory counter --
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # -- Training --
    model.train()
    t_start = time.time()
    patience_counter = 0
    best_test_acc = -1
    best_epoch = 0
    test_history = {}

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            if not is_cached:
                images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        if use_ddp and world_size > 1:
            totals = torch.tensor(
                [running_loss, float(correct), float(total)],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            running_loss, correct, total = totals.tolist()

        train_acc = 100.0 * correct / max(total, 1.0)
        avg_loss = running_loss / max(total, 1.0)
        if rank == 0:
            print(f"    Ep {epoch:2d}/{epochs} -- Loss: {avg_loss:.4f}, Train: {train_acc:.1f}%", end="")

        if epoch % test_every == 0:
            model.eval()
            tc, tt = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    if not is_cached:
                        images, labels = images.to(device), labels.to(device)
                    tc += (model(pixel_values=images).logits.argmax(1) == labels).sum().item()
                    tt += labels.size(0)
            if use_ddp and world_size > 1:
                test_totals = torch.tensor(
                    [float(tc), float(tt)],
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(test_totals, op=dist.ReduceOp.SUM)
                tc, tt = test_totals.tolist()

            test_acc = 100.0 * tc / max(tt, 1.0)
            test_history[epoch] = test_acc
            if rank == 0:
                print(f" | Test: {test_acc:.1f}%", end="")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                patience_counter = 0
            # else:
            #     patience_counter += 1
            #     if patience_counter >= 5:
            #         if rank == 0:
            #             print(f"  STOP (best={best_test_acc:.1f}% @ep{best_epoch})")
            #         break
            model.train()

        if rank == 0:
            print()

        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            if rank == 0:
                print(f"  STOP (reached epoch cap {stop_after_epoch}/{epochs})")
            break

    train_time = time.time() - t_start

    # -- Final evaluation --
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_infer_start = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            if not is_cached:
                images, labels = images.to(device), labels.to(device)
            model(pixel_values=images)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_time = time.time() - t_infer_start

    peak_mem_mb = 0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    if use_ddp and world_size > 1:
        peak_mem = torch.tensor([peak_mem_mb], dtype=torch.float64, device=device)
        dist.all_reduce(peak_mem, op=dist.ReduceOp.MAX)
        peak_mem_mb = float(peak_mem.item())

    if rank == 0:
        print(f"    Best: {best_test_acc:.1f}% @ep{best_epoch} | Train: {train_time:.1f}s | "
              f"Infer: {infer_time:.2f}s | PeakMem: {peak_mem_mb:.0f}MB")

    return {
        "accuracy": best_test_acc,
        "best_epoch": best_epoch,
        "train_time": train_time,
        "infer_time": infer_time,
        "peak_memory_mb": peak_mem_mb,
        "test_history": test_history,
    }


# =============================================================================
# Measurement Utilities
# =============================================================================
def measure_flops(model, image_size=224, device=None):
    """Measure FLOPs for a single forward pass using fvcore."""
    if device is None:
        device = next(model.parameters()).device
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(1, 3, image_size, image_size).to(device)
        model.eval()
        flops = FlopCountAnalysis(model, (dummy,))
        total_flops = flops.total()
        return total_flops
    except Exception as e:
        print(f"  FLOPs measurement failed: {e}")
        return -1


def measure_model_stats(model, config, method_name):
    """Compute all non-accuracy metrics for a model."""
    trainable = count_trainable_params(model)
    total = count_total_params(model)
    flops = measure_flops(model, config["image_size"])

    stats = {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100.0 * trainable / total if total > 0 else 0,
        "flops": flops,
    }
    if flops > 0:
        print(f"  [{method_name}] Trainable: {trainable:,} ({stats['trainable_pct']:.2f}%) | "
              f"FLOPs: {flops/1e9:.2f}G")
    else:
        print(f"  [{method_name}] Trainable: {trainable:,} ({stats['trainable_pct']:.2f}%)")
    return stats
