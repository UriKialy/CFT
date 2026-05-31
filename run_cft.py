#!/usr/bin/env python3
"""
Run CFT (Circuit Fine-Tuning) on VTAB-1K tasks.

Usage:
    python run_cft.py                          # all 19 tasks
    python run_cft.py --tasks cifar dtd        # specific tasks
    python run_cft.py --epochs 40              # override epochs
    python run_cft.py --epochs 100 --stop-after-epoch 50
    python run_cft.py --budget 20              # param budget %
"""
import argparse
import gc
import json
import os
import traceback

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import ViTForImageClassification
from transformers.utils import logging as transformers_logging

try:
    from huggingface_hub.utils import disable_progress_bars as hf_disable_progress_bars
except Exception:
    hf_disable_progress_bars = None

from config import CONFIG, CFT_TASK_CONFIGS, VTAB_TASKS, setup_environment
from dataset import load_vtab_task
from Utils import build_model
from training import train_and_evaluate, measure_model_stats
from circuit_discovery import pretrain_classifier, discover_circuits_eap_ig, select_nodes_by_param_budget


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


def build_cft_method_name(method_tag, metric, pretrain_head, corruption, score_norm, method):
    """Build a stable method name that captures the discovery metric option set."""
    if method_tag:
        return method_tag
    parts = ["cft", metric, method]
    if pretrain_head:
        parts.append("pretrain_head")
    if corruption != "patch_shuffle":
        parts.append(f"corr_{corruption}")
    if score_norm != "param_count":
        parts.append(f"norm_{score_norm}")
    return "__".join(parts)


def run_vit(tasks=None, config=None, use_ddp=False, rank=0, local_rank=0, world_size=1,
            metric="log_prob_diff", pretrain_head=False, corruption="patch_shuffle",
            score_norm="param_count", method="eap-ig", method_tag=None,
            stop_after_epoch=None):
    if config is None:
        config = CONFIG.copy()
    if tasks is None:
        tasks = VTAB_TASKS
    method_name = build_cft_method_name(
        method_tag=method_tag,
        metric=metric,
        pretrain_head=pretrain_head,
        corruption=corruption,
        score_norm=score_norm,
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
            cft_base = ViTForImageClassification.from_pretrained(config["model_name"])
            log_checkpoint_load_event(rank, stage="discovery", event="END")
            log_checkpoint_info(cft_base, config, rank, stage="discovery")
            cft_base.classifier = nn.Linear(cft_base.config.hidden_size, num_classes)
            nn.init.zeros_(cft_base.classifier.bias)
            cft_base = cft_base.to(device)

            # ----- Discovery cache (budget-INDEPENDENT scores + head_state) -----
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "cache", "discovery")
            os.makedirs(cache_dir, exist_ok=True)
            _ph = "ph1" if pretrain_head else "ph0"
            cache_key = f"{task_name}__{corruption}__{score_norm}__{metric}__{method}__{_ph}"
            cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

            if os.path.exists(cache_path):
                print(f"  [discovery cache HIT] {cache_key}")
                blob = torch.load(cache_path, map_location="cpu", weights_only=False)
                circuit_info = blob["circuit_info"]
                head_state = blob["head_state"]
            else:
                if pretrain_head:
                    pretrain_classifier(cft_base, train_ds, device)
                cft_base.eval()
                circuit_info = discover_circuits_eap_ig(
                    cft_base, train_ds, config, device,
                    metric=metric, corruption=corruption, score_norm=score_norm,
                    method=method,
                )
                head_state = None
                if pretrain_head:
                    head_state = {k: v.cpu().clone() for k, v in cft_base.classifier.state_dict().items()}
                torch.save({"circuit_info": circuit_info, "head_state": head_state}, cache_path)
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
                "head_state": head_state,
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
        head_state = circuit_payload.get("head_state")

        # Train CFT
        if is_main_process(rank):
            print(f"\n  -- cft --")

        try:
            log_checkpoint_load_event(rank, stage="train", event="START")
            model = build_model(
                "cft", num_classes, config, device,
                selected_nodes=selected_nodes,
                nodes_map=circuit_info["nodes_map"],
            )
            if head_state is not None:
                model.classifier.load_state_dict(head_state)
                if is_main_process(rank):
                    print("  Loaded pretrained classifier head into training model")
            log_checkpoint_load_event(rank, stage="train", event="END")
            base_model = model.module if hasattr(model, "module") else model
            log_checkpoint_info(base_model, config, rank, stage="train")
            stats = measure_model_stats(model, config, "cft") if is_main_process(rank) else {}
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
            results.update(stats)
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
# =============================================================================
# SWIN / GEMMA ORCHESTRATORS — NEW dispatch glue (the only substantial
# new code in this repo; everything else is copied verbatim from working
# source files).
# =============================================================================
# =============================================================================

def run_swin(tasks=None, config=None, **_unused):
    """Run CFT on Swin (HuggingFace Swinv2) for VTAB-1K / CBIS-DDSM.

    Uses Swin / circuit_discovery_swin / training_swin (all extracted
    verbatim from Swin_vtab1k_CFT.ipynb).
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
        cft_base = M.build_model("full_finetune", num_classes, config, task_name=task_name)
        cft_base = cft_base.to(device).eval()
        circuit_info = D.discover_circuits_eap_ig(cft_base, train_ds, config)
        backbone_params = sum(p.numel() for n, p in cft_base.named_parameters()
                              if "classifier" not in n)
        selected_nodes, used = D.select_nodes_by_param_budget(
            circuit_info["sorted_nodes"], circuit_info["nodes_map"],
            backbone_params, config["cft_param_budget"], task_name=task_name)
        del cft_base; torch.cuda.empty_cache()

        # 2) Build CFT model with selected circuits and train
        model = M.build_model("cft", num_classes, config,
                              selected_nodes=selected_nodes,
                              nodes_map=circuit_info["nodes_map"],
                              task_name=task_name).to(device)
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        result = T.train_and_evaluate(model, train_ds, test_ds, config,
                                      method_name="cft", task_name=task_name,
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
    """Run CFT on Gemma-3-4B-IT for CUB-200 (the notebook's only task).

    Uses Gemma / circuit_discovery_gemma / training_gemma / gemma_utils
    (all extracted verbatim from CFT_Gemma3_4B_IT_CUB200.ipynb).

    NOTE: The notebook code references several module-level globals
    (processor, model, CONFIG, TASK_CLASS_NAMES, STRUCTURED_TASK_CONFIG).
    We inject them into the relevant modules before calling their functions.
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

        # 1) Zero-shot eval to build the confusion matrix → most-confused class map
        print(f"[Gemma] Zero-shot eval on test set to build confusion matrix...")
        zs_acc, confusion = GU.evaluate_zero_shot(test_ds, task_name, return_confusion=True)
        most_confused = GU.get_most_confused_class(confusion, num_classes)
        print(f"[Gemma] Zero-shot acc: {zs_acc:.1f}%")

        # 2) Circuit discovery (EAP-IG) using clean/CF pairs from most_confused
        circuit_info = DG.discover_circuits_eap_ig(model, train_ds, task_name,
                                                    config, most_confused)
        # 3) Select circuits by parameter budget
        selected_nodes, used_params = DG.select_nodes_by_param_budget(
            circuit_info["sorted_nodes"], circuit_info["nodes_map"],
            TOTAL_PARAMS, config["cft_param_budget"])

        # 4) Apply CFT mask and train_generative
        # The notebook's apply_cft references `used_params` as a notebook-level
        # global — inject it into Gemma before calling apply_cft.
        MG.used_params = used_params
        model_cft = MG.apply_cft(model, selected_nodes, circuit_info["nodes_map"])
        model_cft.gradient_checkpointing_enable()

        lr = config.get("learning_rate", 5e-5)
        epochs = config.get("num_epochs", 10)
        _, best_acc = TG.train_generative(model_cft, train_ds, test_ds, task_name,
                                          config, "cft", epochs, lr)
        all_results[task_name] = {"zero_shot": zs_acc, "cft_acc": best_acc}
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
    parser.add_argument("--metric", choices=["log_prob_diff", "logit_diff", "cross_entropy"], default="log_prob_diff",
                        help="Scoring metric for circuit discovery: log_prob_diff | logit_diff | cross_entropy")
    parser.add_argument("--pretrain-head", action="store_true", default=False,
                        help="Pre-train classifier head before discovery")
    parser.add_argument("--corruption",
                        choices=["patch_shuffle", "gaussian", "channel_shuffle",
                                 "intensity_invert", "cutout", "multi", "multi_med"],
                        default="patch_shuffle",
                        help="Corruption for EAP-IG. multi=avg all; multi_med=mammo-friendly subset.")
    parser.add_argument("--score-norm",
                        choices=["param_count", "rank", "mlp_balanced"],
                        default="mlp_balanced",
                        help="param_count | rank | mlp_balanced (default — only MLP scores discounted by mlp_pc/head_pc)")
    parser.add_argument("--method", choices=["eap-ig", "eap"],
                        default="eap-ig",
                        help="Discovery method: eap-ig (IG path) | eap (single gradient)")
    parser.add_argument("--method-tag", type=str, default=None,
                        help="Optional label saved in result field 'method' and run history key")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override CFT_TASK_CONFIGS[task]['lr']")
    parser.add_argument("--wd", type=float, default=None,
                        help="Override CFT_TASK_CONFIGS[task]['wd']")
    parser.add_argument("--label-smoothing", type=float, default=None,
                        help="Override CFT_TASK_CONFIGS[task]['label_smoothing']")
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
    for tn in (args.tasks or VTAB_TASKS):
        if tn in CFT_TASK_CONFIGS:
            if args.lr is not None: CFT_TASK_CONFIGS[tn]["lr"] = args.lr
            if args.wd is not None: CFT_TASK_CONFIGS[tn]["wd"] = args.wd
            if args.label_smoothing is not None: CFT_TASK_CONFIGS[tn]["label_smoothing"] = args.label_smoothing

    selected_metric = args.metric

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

    # Make the per-task HPs visible to the backbone-specific runners that look
    # up CFT_TASK_CONFIGS by module name. (run_vit reads the top-level
    # CFT_TASK_CONFIGS already imported above.)
    if args.backbone != "vit":
        # Replace contents of CFT_TASK_CONFIGS in-place
        CFT_TASK_CONFIGS.clear()
        CFT_TASK_CONFIGS.update(backbone_task_configs)

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
                metric=selected_metric,
                pretrain_head=args.pretrain_head,
                corruption=args.corruption,
                score_norm=args.score_norm,
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
