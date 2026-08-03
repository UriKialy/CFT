"""
Swin training loop — CFT only.
Mirror of training.py (ViT) with Swin's pixel_values call signature.
"""
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from dataset import GPUCachedDataset


def measure_model_stats(model, *_ignored, **_ignored_kw):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def train_and_evaluate(model, train_ds, test_ds, config,
                       method_name="cft", task_name="", scaler=None,
                       device=None, use_ddp=False, rank=0, world_size=1,
                       stop_after_epoch=None, cft_task_configs=None,
                       **_unused):
    """
    Swin CFT training: per-epoch test eval, best-acc tracking.
    Returns dict mirroring training.py (ViT).
    """
    if device is None:
        device = next(model.parameters()).device

    if cft_task_configs is None:
        from config import CFT_TASK_CONFIGS
        cft_task_configs = CFT_TASK_CONFIGS

    task_cfg = cft_task_configs.get(task_name, {})
    lr              = task_cfg.get("lr", config.get("learning_rate", 1e-4))
    wd              = task_cfg.get("wd", config.get("weight_decay", 0.01))
    label_smoothing = task_cfg.get("label_smoothing", 0.0)
    batch_size      = task_cfg.get("batch_size", config.get("batch_size", 64))
    epochs     = config.get("num_epochs", 100)
    run_epochs = task_cfg.get("stop_after", epochs)
    if stop_after_epoch is not None:
        run_epochs = min(run_epochs, stop_after_epoch)

    # Train loader
    train_is_cached = isinstance(train_ds, GPUCachedDataset)
    train_sampler = None
    if use_ddp and world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                           rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              batch_size=batch_size,
                              num_workers=0 if train_is_cached else 4,
                              pin_memory=not train_is_cached)

    # split CFT-masked params (no weight decay) from the rest
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    masked_params = getattr(base_model, "_cft_no_weight_decay_params", None)
    if masked_params:
        masked_ids = {id(p) for p in masked_params if p.requires_grad}
        decay_params    = [p for p in trainable_params if id(p) not in masked_ids]
        no_decay_params = [p for p in trainable_params if id(p) in masked_ids]
        groups = []
        if decay_params:
            groups.append({"params": decay_params,    "weight_decay": wd})
        if no_decay_params:
            groups.append({"params": no_decay_params, "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(groups, lr=lr)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

    total_steps  = epochs * len(train_loader)
    warmup_steps = 0
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    test_is_cached = isinstance(test_ds, GPUCachedDataset)
    test_sampler = None
    if use_ddp and world_size > 1:
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size,
                                          rank=rank, shuffle=False)
    test_loader = DataLoader(test_ds, shuffle=False, sampler=test_sampler,
                             batch_size=batch_size,
                             num_workers=0 if test_is_cached else 4,
                             pin_memory=not test_is_cached)

    t_start = time.time()
    best_test_acc = -1.0
    best_epoch = 0

    for epoch in range(1, run_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            if not train_is_cached:
                images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values=images).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total   += labels.size(0)

        if use_ddp and world_size > 1:
            tot = torch.tensor([running_loss, float(correct), float(total)],
                               dtype=torch.float64, device=device)
            dist.all_reduce(tot, op=dist.ReduceOp.SUM)
            running_loss, correct, total = tot.tolist()

        train_acc = 100.0 * correct / max(total, 1.0)
        avg_loss  = running_loss / max(total, 1.0)

        model.eval()
        tc, tt = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                if not test_is_cached:
                    images, labels = images.to(device), labels.to(device)
                tc += (model(pixel_values=images).logits.argmax(1) == labels).sum().item()
                tt += labels.size(0)
        if use_ddp and world_size > 1:
            test_totals = torch.tensor([float(tc), float(tt)],
                                       dtype=torch.float64, device=device)
            dist.all_reduce(test_totals, op=dist.ReduceOp.SUM)
            tc, tt = test_totals.tolist()
        test_acc = 100.0 * tc / max(tt, 1.0)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

        if rank == 0:
            print(f"    Ep {epoch:2d}/{run_epochs} -- Loss: {avg_loss:.4f}, "
                  f"Train: {train_acc:.1f}% | Test: {test_acc:.1f}%")

    train_time = time.time() - t_start

    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_infer_start = time.time()
    tc, tt = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            if not test_is_cached:
                images, labels = images.to(device), labels.to(device)
            tc += (model(pixel_values=images).logits.argmax(1) == labels).sum().item()
            tt += labels.size(0)
    if use_ddp and world_size > 1:
        test_totals = torch.tensor([float(tc), float(tt)],
                                   dtype=torch.float64, device=device)
        dist.all_reduce(test_totals, op=dist.ReduceOp.SUM)
        tc, tt = test_totals.tolist()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_time = time.time() - t_infer_start
    final_test_acc = 100.0 * tc / max(tt, 1.0)

    peak_mem_mb = 0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        if use_ddp and world_size > 1:
            peak = torch.tensor([peak_mem_mb], dtype=torch.float64, device=device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            peak_mem_mb = float(peak.item())

    if rank == 0:
        print(f"    ✓ Best: {best_test_acc:.1f}% @ep{best_epoch} | "
              f"Final: {final_test_acc:.1f}% | "
              f"Train: {train_time:.1f}s | Infer: {infer_time:.2f}s | PeakMem: {peak_mem_mb:.0f}MB")

    return {
        "accuracy":       best_test_acc,
        "final_accuracy": final_test_acc,
        "best_epoch":     best_epoch,
        "epochs_run":     run_epochs,
    }
