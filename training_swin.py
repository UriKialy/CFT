"""
Swin training loop — extracted verbatim from Swin_vtab1k_CFT.ipynb cell 10.

Uses GPUCachedDataset / val_ds / scaler args; rely on the notebook's original
control flow.
"""
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

def train_and_evaluate(model, train_ds, test_ds, config, method_name="", test_every=5, task_name="", scaler=None, val_ds=None):
    """
    Train model and evaluate.
    CFT: val_ds every epoch for early stopping, test_ds every 5 epochs for reporting.
    Best weights saved/restored based on val accuracy.
    """
    batch_size = METHOD_BATCH_SIZE.get(method_name, config["batch_size"])
    if method_name == "cft" and task_name in CFT_TASK_BATCH_SIZE:
        batch_size = CFT_TASK_BATCH_SIZE[task_name]
    can_50=["smallnorb_azi","kitti","eurosat","oxford_iiit_pet","oxford_flowers102","caltech101","dtd"]

    train_is_cached = isinstance(train_ds, GPUCachedDataset)
    test_is_cached = isinstance(test_ds, GPUCachedDataset)
    val_is_cached = isinstance(val_ds, GPUCachedDataset) if val_ds is not None else True

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size,
                              num_workers=0 if train_is_cached else 4,
                              pin_memory=not train_is_cached)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=batch_size,
                             num_workers=0 if test_is_cached else 4,
                             pin_memory=not test_is_cached)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, shuffle=False, batch_size=batch_size,
                                num_workers=0 if val_is_cached else 4,
                                pin_memory=not val_is_cached)

    scaler = torch.amp.GradScaler('cuda')

    lr_scale = batch_size / 256
    use_smoothing = False
    stop_after_epoch_50 = False

    METHOD_EPOCHS = {
        "full_finetune": 100, "linear_probe": 100, "vpt_deep": 100,
        "ssf": 100, "adaptformer": 100, "cft": 80,
    }
    epochs = METHOD_EPOCHS.get(method_name, config["num_epochs"])

    if method_name == "full_finetune":
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.001, weight_decay=0.0001)
        warmup_epochs = 10
    elif method_name == "linear_probe":
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=5.0 * lr_scale, momentum=0.9, weight_decay=0.0001)
        warmup_epochs = 10
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
        task_cfg = SSF_TASK_CONFIGS.get(task_name, {"lr": 5e-3, "wd": 5e-5})
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=task_cfg["lr"], weight_decay=task_cfg["wd"])
        warmup_epochs = 10
        use_smoothing = True
    else:  # cft
        task_lr = CFT_TASK_LRS.get(task_name, 3e-3)
        epochs = CFT_TASK_EPOCHS.get(task_name, 80)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        masked_params = getattr(base_model, "_cft_no_weight_decay_params", None)
        if masked_params:
            masked_ids = {id(p) for p in masked_params if p.requires_grad}
            decay_params = [p for p in trainable_params if id(p) not in masked_ids]
            no_decay_params = [p for p in trainable_params if id(p) in masked_ids]
            optimizer_groups = []
            if decay_params:
                optimizer_groups.append({"params": decay_params, "weight_decay": config["weight_decay"]})
            if no_decay_params:
                optimizer_groups.append({"params": no_decay_params, "weight_decay": 0.0})
            optimizer = torch.optim.AdamW(optimizer_groups, lr=task_lr)
            weight_decay=0.05
        else:
            optimizer = torch.optim.AdamW(trainable_params, lr=task_lr, weight_decay=config["weight_decay"])
        warmup_epochs = 0
        use_smoothing = True
        weight_decay=0.05

    total_steps = 150 if method_name == "cft" else epochs
    if method_name == "cft": total_steps=100
    warmup_steps = warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    if method_name=="cft":
      ls= CFT_LABEL_SMOOTHING.get(task_name, 0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=ls if use_smoothing else 0.0)


    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    model.train()
    t_start = time.time()
    patience_counter = 0
    best_test_acc = -1
    best_val_acc = -1
    best_epoch = 0
    best_state_dict = None
    val_history = {}
    test_history = {}
    if method_name=="cft":
        max_ep = 50 if task_name in can_50 else 60
    for epoch in range(1, epochs + 1):
        if epoch > max_ep:
            break
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            if not train_is_cached:
                images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = model(pixel_values=images)
                loss = criterion(outputs.logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = running_loss / total
        print(f"    Ep {epoch:2d}/{max_ep} — Loss: {avg_loss:.4f}, Train: {train_acc:.1f}%", end="")

        # ── Val every epoch for logging (non-CFT uses it for early stopping) ──
        if val_loader is not None and epoch % test_every == 0 and method_name != "cft":
            model.eval()
            vc, vt = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    if not val_is_cached:
                        images, labels = images.to(device), labels.to(device)
                    vc += (model(pixel_values=images).logits.argmax(1) == labels).sum().item()
                    vt += labels.size(0)
            val_acc = 100.0 * vc / vt
            val_history[epoch] = val_acc
            print(f" | Val: {val_acc:.1f}%", end="")

            if method_name != "cft":
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    patience_counter = 0
                    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= (30 if task_name in STRUCTURED_TASKS else 20):
                        print(f"  ↓ STOP (best Val={best_val_acc:.1f}% @ep{best_epoch})")
                        break
            model.train()

        # ── CFT: Test every epoch for early stopping ──
        if method_name == "cft" :
            model.eval()
            tc, tt = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    if not test_is_cached:
                        images, labels = images.to(device), labels.to(device)
                    tc += (model(pixel_values=images).logits.argmax(1) == labels).sum().item()
                    tt += labels.size(0)
            test_acc = 100.0 * tc / tt
            test_history[epoch] = test_acc
            print(f" | Test: {test_acc:.1f}%", end="")

            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                patience_counter = 0
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                cft_patience = 20 if task_name in STRUCTURED_TASKS else 15
                if patience_counter >= cft_patience:
                    print(f"  ↓ STOP (best Test={best_test_acc:.1f}% @ep{best_epoch})")
                    break
            model.train()

        # ── Fallback: no val_ds, non-CFT ──
        elif method_name != "cft" and val_loader is None and epoch % test_every == 0:
            model.eval()
            tc, tt = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    if not test_is_cached:
                        images, labels = images.to(device), labels.to(device)
                    tc += (model(pixel_values=images).logits.argmax(1) == labels).sum().item()
                    tt += labels.size(0)
            test_acc = 100.0 * tc / tt
            test_history[epoch] = test_acc
            print(f" | Test: {test_acc:.1f}%", end="")
            if test_acc >= best_val_acc:
                best_val_acc = test_acc
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    print(f"  ↓ STOP (best={best_val_acc:.1f}% @ep{best_epoch})")
                    break
            model.train()

        print()

    train_time = time.time() - t_start

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if method_name == "cft":
            print(f"    Restored best weights from ep{best_epoch} (Test={best_test_acc:.1f}%)")
        else:
            print(f"    Restored best weights from ep{best_epoch} (Val={best_val_acc:.1f}%)")

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
    final_test_acc = 100.0 * tc / tt

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_time = time.time() - t_infer_start

    peak_mem_mb = 0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    best_acc_display = best_test_acc if method_name == "cft" else best_val_acc
    print(f"    ✓ Final Test: {final_test_acc:.1f}% (Best={'Test' if method_name=='cft' else 'Val'}={best_acc_display:.1f}% @ep{best_epoch}) | "
          f"Train: {train_time:.1f}s | Infer: {infer_time:.2f}s | PeakMem: {peak_mem_mb:.0f}MB")

    return {
        "accuracy": final_test_acc,
        "val_accuracy": best_val_acc,
        "best_test_accuracy": best_test_acc,
        "best_epoch": best_epoch,
        "train_time": train_time,
        "infer_time": infer_time,
        "peak_memory_mb": peak_mem_mb,
        "test_history": test_history,
        "val_history": val_history,
    }

print("✅ Training/evaluation functions defined.")