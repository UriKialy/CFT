#!/usr/bin/env python3
"""
Run CFT (Circuit Fine-Tuning).
"""
import argparse
import gc
import json
import os
import traceback

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.utils import logging as transformers_logging

try:
    from huggingface_hub.utils import disable_progress_bars as hf_disable_progress_bars
except Exception:
    hf_disable_progress_bars = None

from config import CONFIG, CFT_TASK_CONFIGS, SWIN_TASK_CONFIGS, VTAB_TASKS, setup_environment
from dataset import load_vtab_task
from Utils import build_model
from training import train_and_evaluate
from circuit_discovery import discover_circuits_eap_ig, select_nodes_by_param_budget

def disable_hf_progress_output():
    """Disable Hugging Face progress bars for cleaner logs."""
    if hf_disable_progress_bars is not None:
        hf_disable_progress_bars()
    if hasattr(transformers_logging, "disable_progress_bar"):
        transformers_logging.disable_progress_bar()

def init_ddp(use_ddp=False):
    """Initialize DDP from torchrun environment variables."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    should_use_ddp = use_ddp or world_size > 1
    rank = 0
    local_rank = 0

    if should_use_ddp and world_size > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        should_use_ddp = False

    return should_use_ddp, rank, local_rank, world_size

def destroy_ddp(use_ddp=False):
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    return rank == 0

def log_checkpoint_info(model, config, rank, stage):
    """Print which checkpoint ID was requested and what was actually loaded."""
    if not is_main_process(rank):
        return
    requested = config.get("model_name", "unknown")
    loaded = getattr(model.config, "_name_or_path", "unknown")
    print(f"[{stage}] checkpoint requested: {requested}")
    print(f"[{stage}] checkpoint loaded:    {loaded}")

def log_checkpoint_load_event(rank, stage, event):
    """Print explicit start/end events for checkpoint loading."""
    if is_main_process(rank):
        print(f"[{stage}] {event} loading weights...")

def make_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def build_cft_method_name(method_tag, corruption, method):
    """Build a stable method name from the discovery option set."""
    if method_tag:
        return method_tag
    parts = ["cft", "log_prob_diff", method]
    if corruption != "patch_shuffle":
        parts.append(f"corr_{corruption}")
    return "__".join(parts)

def run_vit(tasks=None, config=None, use_ddp=False, rank=0, local_rank=0, world_size=1,
            corruption="patch_shuffle",
            method="eap-ig", method_tag=None,
            stop_after_epoch=None):
    if config is None:
        config = CONFIG.copy()
    if tasks is None:
        tasks = VTAB_TASKS
    method_name = build_cft_method_name(
        method_tag=method_tag,
        corruption=corruption,
        method=method,
    )

    if use_ddp and world_size > 1:
        original_batch_size = config["batch_size"]
        config["batch_size"] = max(1, original_batch_size // world_size)
        config["use_gpu_cache"] = False
        if is_main_process(rank):
            print(
                f"DDP config override: batch_size {original_batch_size} -> "
                f"{config['batch_size']} (per GPU), use_gpu_cache=False"
            )

    setup_environment(config)
    if use_ddp and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config["seed"] + rank)
    np.random.seed(config["seed"] + rank)

    if is_main_process(rank):
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(device)}")
        if use_ddp:
            print(f"DDP enabled: world_size={world_size}")

    all_results = {}

    for task_idx, task_name in enumerate(tasks):
        if is_main_process(rank):
            print(f"\n{'#'*70}")
            print(f"# TASK {task_idx+1}/{len(tasks)}: {task_name}")
            print(f"{'#'*70}")

        train_ds, test_ds, num_classes = load_vtab_task(task_name, config, device)

        # Circuit discovery
        payload = [None]
        if is_main_process(rank):
            print(f"\n  Circuit discovery for {task_name}...")
            log_checkpoint_load_event(rank, stage="discovery", event="START")
            # build_model with selected_nodes=None -> near-zero head init for discovery
            cft_base = build_model(num_classes, config, device)
            log_checkpoint_load_event(rank, stage="discovery", event="END")
            log_checkpoint_info(cft_base, config, rank, stage="discovery")

            # ----- Discovery cache (budget-INDEPENDENT) -----
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "cache", "discovery")
            os.makedirs(cache_dir, exist_ok=True)
            cache_key = f"{task_name}__{corruption}__log_prob_diff__{method}"
            cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

            if os.path.exists(cache_path):
                print(f"  [discovery cache HIT] {cache_key}")
                blob = torch.load(cache_path, map_location="cpu", weights_only=False)
                circuit_info = blob["circuit_info"]
            else:
                cft_base.eval()
                circuit_info = discover_circuits_eap_ig(
                    cft_base, train_ds, config, device,
                    corruption=corruption, method=method,
                )
                torch.save({"circuit_info": circuit_info}, cache_path)
                print(f"  [discovery cache SAVE] {cache_key}")

            # ----- Selection (budget-DEPENDENT, always runs) -----
            backbone_params = sum(
                p.numel() for n, p in cft_base.named_parameters() if "classifier" not in n
            )
            selected_nodes, used_params = select_nodes_by_param_budget(
                circuit_info["sorted_nodes"], circuit_info["nodes_map"],
                backbone_params, config["cft_param_budget"],
            )
            payload[0] = {
                "circuit_info": circuit_info,
                "backbone_params": backbone_params,
                "selected_nodes": selected_nodes,
                "used_params": used_params,
            }
            del cft_base
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if use_ddp:
            dist.broadcast_object_list(payload, src=0)
            dist.barrier()

        circuit_payload = payload[0]
        circuit_info = circuit_payload["circuit_info"]
        backbone_params = circuit_payload["backbone_params"]
        selected_nodes = circuit_payload["selected_nodes"]
        used_params = circuit_payload["used_params"]

        # Train CFT
        if is_main_process(rank):
            print(f"\n  -- cft --")

        try:
            log_checkpoint_load_event(rank, stage="train", event="START")
            model = build_model(
                num_classes, config, device,
                selected_nodes=selected_nodes,
                nodes_map=circuit_info["nodes_map"],
            )
            log_checkpoint_load_event(rank, stage="train", event="END")
            base_model = model.module if hasattr(model, "module") else model
            log_checkpoint_info(base_model, config, rank, stage="train")
            if use_ddp:
                if device.type == "cuda":
                    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
                else:
                    model = DDP(model)

            results = train_and_evaluate(
                model, train_ds, test_ds, config, "cft",
                task_name=task_name, device=device,
                cft_task_configs=CFT_TASK_CONFIGS,
                use_ddp=use_ddp, rank=rank, world_size=world_size,
                stop_after_epoch=stop_after_epoch,
            )
            results["method"] = method_name
            results["task"] = task_name
            results["circuit_info"] = {
                "selected_nodes": list(selected_nodes),
                "used_params": used_params,
                "backbone_params": backbone_params,
            }
            if is_main_process(rank):
                all_results[task_name] = results
                print(f"    Best: {results['accuracy']:.1f}% @epoch {results['best_epoch']}")

        except Exception as e:
            if is_main_process(rank):
                print(f"    FAILED: {e}")
                traceback.print_exc()
                all_results[task_name] = {"method": method_name, "task": task_name, "error": str(e)}

        finally:
            if "model" in dir():
                del model
            del train_ds, test_ds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save
    if is_main_process(rank):
        results_path = os.path.join(config["save_dir"], "cft_results.json")
        current_run_results = {
            task: {k: make_serializable(v) for k, v in res.items()}
            for task, res in all_results.items()
        }

        existing_runs = {}
        if os.path.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    if isinstance(existing.get("__runs__"), dict):
                        existing_runs.update(existing["__runs__"])
                    else:
                        legacy_results = {
                            k: v for k, v in existing.items()
                            if isinstance(k, str) and not k.startswith("__")
                        }
                        if legacy_results:
                            existing_runs["legacy"] = legacy_results
            except Exception:
                # If old file is malformed, start a fresh multi-run container.
                existing_runs = {}

        run_key = method_name
        if run_key in existing_runs:
            suffix = 2
            while f"{method_name}#{suffix}" in existing_runs:
                suffix += 1
            run_key = f"{method_name}#{suffix}"
        existing_runs[run_key] = current_run_results

        serializable = dict(current_run_results)
        serializable["__runs__"] = existing_runs
        serializable["__last_run__"] = run_key
        serializable["__last_method__"] = method_name
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\nResults saved to {results_path} (run key: {run_key})")

    # Summary
    if is_main_process(rank):
        accs = [r["accuracy"] for r in all_results.values() if "accuracy" in r]
        if accs:
            print(f"\nCFT Summary: {len(accs)} tasks, mean accuracy: {sum(accs)/len(accs):.1f}%")
            for task, res in all_results.items():
                if "accuracy" in res:
                    print(f"  {task:<25s} {res['accuracy']:6.1f}%")

    return all_results

# =============================================================================
# SWIN / GEMMA ORCHESTRATORS — NEW dispatch glue 
# =============================================================================

def run_swin(tasks=None, config=None, **_unused):
    """Run CFT on Swin for VTAB-1K / CBIS-DDSM.

    Uses Swin / circuit_discovery_swin / training_swin
    """
    import Swin as M
    import circuit_discovery_swin as D
    import training_swin as T
    from dataset import load_vtab_task

    if config is None:
        config = CONFIG.copy()
    if tasks is None:
        tasks = VTAB_TASKS

    setup_environment(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Swin] Device: {device}")

    all_results = {}
    for task_idx, task_name in enumerate(tasks):
        print(f"\n{'#'*70}\n# TASK {task_idx+1}/{len(tasks)}: {task_name} [Swin/CFT]\n{'#'*70}")
        train_ds, test_ds, num_classes = load_vtab_task(task_name, config, device)

        # 1) Circuit discovery
        cft_base = M.build_model(num_classes, config, task_name=task_name)
        cft_base = cft_base.to(device).eval()
        circuit_info = D.discover_circuits_eap_ig(cft_base, train_ds, config)
        backbone_params = sum(p.numel() for n, p in cft_base.named_parameters()
                              if "classifier" not in n)
        selected_nodes, used = D.select_nodes_by_param_budget(
            circuit_info["sorted_nodes"], circuit_info["nodes_map"],
            backbone_params, config["cft_param_budget"], task_name=task_name)
        del cft_base; torch.cuda.empty_cache()

        # 2) Build CFT model with selected circuits and train
        model = M.build_model(num_classes, config,
                              selected_nodes=selected_nodes,
                              nodes_map=circuit_info["nodes_map"],
                              task_name=task_name).to(device)
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        result = T.train_and_evaluate(model, train_ds, test_ds, config,
                                      method_name="cft", task_name=task_name,
                                      cft_task_configs=SWIN_TASK_CONFIGS,
                                      scaler=scaler)
        all_results[task_name] = result
        # Save incremental JSON
        os.makedirs(config["save_dir"], exist_ok=True)
        with open(os.path.join(config["save_dir"], "swin_cft_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        del model; torch.cuda.empty_cache(); gc.collect()
    print("\n[Swin] Done.")
    return all_results

def run_gemma(tasks=None, config=None, **_unused):
    """Run CFT on Gemma-3-4B-IT for CUB-200.
    Uses Gemma / circuit_discovery_gemma / training_gemma / gemma_utils
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import Gemma as MG
    import circuit_discovery_gemma as DG
    import training_gemma as TG
    import gemma_utils as GU
    from dataset_gemma import _load_cub200, load_cub_class_names

    if config is None:
        config = GEMMA_CONFIG.copy() if 'GEMMA_CONFIG' in globals() else CONFIG.copy()
    if tasks is None:
        tasks = GEMMA_TASKS if 'GEMMA_TASKS' in globals() else ["cub200"]

    setup_environment(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Gemma] Device: {device}")

    # ── Load Gemma processor + model ONCE ──
    model_id = config["model_name"]
    print(f"[Gemma] Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    TOTAL_PARAMS = sum(p.numel() for p in model.parameters())
    print(f"[Gemma] Model loaded ({TOTAL_PARAMS:,} params).")

    # ── Inject globals required by the notebook-extracted modules ──
    cub_names = load_cub_class_names(config["data_dir"])
    TASK_CLASS_NAMES = {"cub200": cub_names}
    STRUCTURED_TASK_CONFIG = {}
    for mod in (GU, DG, TG, MG):
        mod.processor = processor
        mod.model = model
        mod.device = device
        mod.CONFIG = config
        mod.TASK_CLASS_NAMES = TASK_CLASS_NAMES
        mod.STRUCTURED_TASK_CONFIG = STRUCTURED_TASK_CONFIG
        mod.TOTAL_PARAMS = TOTAL_PARAMS

    all_results = {}
    for task_idx, task_name in enumerate(tasks):
        print(f"\n{'#'*70}\n# TASK {task_idx+1}/{len(tasks)}: {task_name} [Gemma/CFT]\n{'#'*70}")
        # Gemma uses CUB-200 (PIL access)
        train_ds, test_ds, num_classes = _load_cub200(config["data_dir"], config)

        # Circuit discovery (EAP-IG). CF pairs use a small hardcoded confusion
        # table inside circuit_discovery_gemma (ZS was ~random anyway).
        circuit_info = DG.discover_circuits_eap_ig(model, train_ds, task_name, config)
        # 3) Select circuits by parameter budget
        selected_nodes, used_params = DG.select_nodes_by_param_budget(
            circuit_info["sorted_nodes"], circuit_info["nodes_map"],
            TOTAL_PARAMS, config["cft_param_budget"])

        # 4) Apply CFT mask and train_generative
        # global — inject it into Gemma before calling apply_cft.
        MG.used_params = used_params
        model_cft = MG.apply_cft(model, selected_nodes, circuit_info["nodes_map"])
        model_cft.gradient_checkpointing_enable()

        lr = config.get("learning_rate", 5e-5)
        epochs = config.get("num_epochs", 10)
        _, best_acc = TG.train_generative(model_cft, train_ds, test_ds, task_name,
                                          config, "cft", epochs, lr)
        all_results[task_name] = {"cft_acc": best_acc}
        os.makedirs(config["save_dir"], exist_ok=True)
        with open(os.path.join(config["save_dir"], "gemma_cft_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print("\n[Gemma] Done.")
    return all_results

if __name__ == "__main__":
    disable_hf_progress_output()
    parser = argparse.ArgumentParser(description="Run CFT on VTAB-1K")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Task names (default: all 19)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--stop-after-epoch", type=int, default=None,
                        help="Early stop cap. Example: --epochs 100 --stop-after-epoch 50")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Training batch size (global for single GPU, base before DDP scaling)")
    parser.add_argument("--budget", type=float, default=None,
                        help="CFT param budget %%")
    parser.add_argument("--cft-batch-size", type=int, default=None,
                        help="Batch size for CFT circuit discovery (EAP-IG)")
    parser.add_argument("--ig-steps", type=int, default=None,
                        help="Integrated gradient steps")
    parser.add_argument("--discovery-pct", type=float, default=None,
                        help="%% of training data for circuit discovery")
    parser.add_argument("--corruption",
                        choices=["patch_shuffle", "gaussian", "channel_shuffle",
                                 "intensity_invert", "cutout", "multi", "multi_med"],
                        default="patch_shuffle",
                        help="Corruption for EAP-IG. multi=avg all; multi_med=mammo-friendly subset.")
    parser.add_argument("--method", choices=["eap-ig", "eap"],
                        default="eap-ig",
                        help="Discovery method: eap-ig (IG path) | eap (single gradient)")
    parser.add_argument("--method-tag", type=str, default=None,
                        help="Optional label saved in result field 'method' and run history key")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override per-task 'lr' for the selected backbone")
    parser.add_argument("--wd", type=float, default=None,
                        help="Override per-task 'wd' for the selected backbone")
    parser.add_argument("--label-smoothing", type=float, default=None,
                        help="Override per-task 'label_smoothing' for the selected backbone")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Classifier-head dropout (sets config['head_dropout'])")
    parser.add_argument("--backbone", choices=["vit", "swin", "gemma"], default="vit",
                        help="Which backbone to fine-tune: vit | swin | gemma")
    parser.add_argument("--dataset", choices=["vtab", "cbis", "cub200"], default="vtab",
                        help="Which dataset: vtab (VTAB-1K) | cbis (CBIS-DDSM) | cub200 (CUB-200, Gemma only)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data_dir (default uses config's value).")
    parser.add_argument("--ddp", action="store_true",
                        help="Enable DistributedDataParallel (launch with torchrun)")
    args = parser.parse_args()

    config = CONFIG.copy()
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.budget is not None:
        config["cft_param_budget"] = args.budget
    if args.cft_batch_size is not None:
        config["cft_batch_size"] = args.cft_batch_size
    if args.ig_steps is not None:
        config["cft_ig_steps"] = args.ig_steps
    if args.discovery_pct is not None:
        config["cft_discovery_pct"] = args.discovery_pct
    if args.dropout is not None:
        config["head_dropout"] = args.dropout


    # Pick the backbone-specific CONFIG and per-task HP table
    from config import get_backbone_config, get_task_configs
    backbone_config = get_backbone_config(args.backbone)
    backbone_task_configs = get_task_configs(args.backbone)
    # Carry over any CLI overrides already applied to `config`
    for k in ("num_epochs", "batch_size", "cft_param_budget", "cft_batch_size",
              "cft_ig_steps", "cft_discovery_pct", "head_dropout"):
        if k in config and config[k] != CONFIG.get(k):
            backbone_config[k] = config[k]
    if args.data_dir is not None:
        backbone_config["data_dir"] = args.data_dir
    config = backbone_config
    # Per-task CLI overrides apply to the backbone's per-task dict.
    for tn in (args.tasks or VTAB_TASKS):
        if tn in backbone_task_configs:
            if args.lr is not None:
                backbone_task_configs[tn]["lr"] = args.lr
            if args.wd is not None:
                backbone_task_configs[tn]["wd"] = args.wd
            if args.label_smoothing is not None:
                backbone_task_configs[tn]["label_smoothing"] = args.label_smoothing


    use_ddp, rank, local_rank, world_size = init_ddp(use_ddp=args.ddp)
    try:
        if args.backbone == "vit":
            run_vit(
                tasks=args.tasks,
                config=config,
                use_ddp=use_ddp,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                corruption=args.corruption,
                method=args.method,
                method_tag=args.method_tag,
                stop_after_epoch=args.stop_after_epoch,
            )
        elif args.backbone == "swin":
            run_swin(tasks=args.tasks, config=config)
        elif args.backbone == "gemma":
            run_gemma(tasks=args.tasks, config=config)
        else:
            raise ValueError(f"Unknown backbone: {args.backbone!r}")
    finally:
        destroy_ddp(use_ddp=use_ddp)
