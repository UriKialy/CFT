# CFT — Circuit Fine-Tuning

Unified command-line implementation of **Circuit Fine-Tuning (CFT)** for three
backbones — **ViT, Swin (v2), and Gemma-3-4B** — on three datasets:
**VTAB-1K, CBIS-DDSM, and CUB-200**.

CFT is a parameter-efficient fine-tuning (PEFT) method that uses
**EAP-IG** (Edge Attribution Patching with Integrated Gradients,
[Hanna et al., COLM 2024](https://arxiv.org/abs/2403.17806))
to identify a small subset of attention heads and MLP blocks that are most
important for a downstream task. Only those discovered "circuit" components are
unfrozen during fine-tuning, giving full-finetune-quality accuracy at a fraction
of the trainable parameter budget.

## What CFT does, end-to-end

For each (backbone, task) pair the pipeline is:

1. **Load** the pretrained backbone.
2. **Discover circuits** — run EAP-IG on a small slice of training data.
   The score of a node (attention head / MLP) is its average gradient-times-difference
   between *clean* and *corrupted* activations along the integrated-gradient path
   from corrupted to clean inputs. Higher score ⇒ more task-relevant.
3. **Select nodes** under a parameter budget (`--budget`, default 17 % of backbone params).
4. **Mask gradients** on the selected nodes only, train the classifier head normally,
   and train for the per-task best-epoch count from `CFT_TASK_CONFIGS`.

Corrupted inputs are produced by patch-shuffling, gaussian noise, or
channel-shuffling (configurable via `--corruption`). For Gemma (a VLM), the
corruption is the same image but with the *most-confused class* swapped in, with
the corresponding text answer — measured from a zero-shot confusion matrix on the
test set.

## Quick start

```bash
# 0. install deps
pip install -r requirements.txt

# 1. download VTAB-1K (about 1 GB, one-time)
bash setup_vtab.sh                  # downloads to ./data/vtab-1k

# 2. run CFT on ViT for all 19 VTAB tasks (uses the best per-task HPs)
python run_cft.py --backbone vit --dataset vtab

# 3. ...or on Swin (HuggingFace Swinv2, auto-downloaded on first run)
python run_cft.py --backbone swin --dataset vtab

# 4. ...or just a few tasks
python run_cft.py --backbone vit --dataset vtab --tasks cifar dtd svhn
```

## Backbones supported

| `--backbone` | HuggingFace model id                              | Notes                                                                                                                                       |
| ------------ | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `vit`        | `google/vit-base-patch16-224-in21k`               | Default. 224 × 224 input.                                                                                                                  |
| `swin`       | `microsoft/swinv2-base-patch4-window8-256`        | 256 × 256. CFT code extracted from `Swin_vtab1k_CFT.ipynb`.                                                                                |
| `gemma`      | `google/gemma-3-4b-it`                            | Vision-language model. Generative fine-tuning. Use **only with** `--dataset cub200`. |

> **HuggingFace gating note.** `google/gemma-3-4b-it` requires you to accept the
> license on its model page and run `huggingface-cli login` (or set `HF_TOKEN`)
> before the first `from_pretrained` call.



## Important CLI flags

```
--backbone {vit, swin, gemma}     # required for non-default (ViT)
--dataset  {vtab, cbis, cub200}   # default vtab
--tasks    <name1 name2 ...>      # subset; default = all tasks for that dataset
--data-dir <path>                 # override default data location
--budget   <float>                # CFT param budget % (default per-task from CFT_TASK_CONFIGS)
--epochs   <int>                  # override num_epochs
--stop-after-epoch <int>          # early-stop cap
--corruption {patch_shuffle, gaussian, channel_shuffle, intensity_invert, cutout, multi, multi_med}
--metric   {log_prob_diff, logit_diff, cross_entropy}
--ig-steps <int>                  # integrated gradient steps (default 12 for ViT, 8 for Gemma)
--discovery-pct <float>           # % of training data for circuit discovery
--lr / --wd / --label-smoothing / --dropout    # per-run overrides
```

Run `python run_cft.py --help` for the full list.

## What's inside

```
config.py                      # CONFIG dicts + per-task best-HP tables
                               #   - top section: ViT (used by default)
                               #   - SWIN_*  block: Swin best HPs (verbatim from notebook)
                               #   - GEMMA_* block: Gemma config (verbatim from notebook)
                               # plus get_backbone_config() / get_task_configs() dispatchers
dataset.py                     # VTAB / CBIS-DDSM loaders (ViT/Swin path)
dataset_gemma.py               # CUB-200 + PIL-access loaders (Gemma path)
methods.py                     # ViT PEFT methods (build_model / apply_cft / ...)
methods_swin.py                # Swin PEFT methods (verbatim from Swin notebook cell 7)
methods_gemma.py               # Gemma apply_cft (verbatim from Gemma notebook cell 14)
circuit_discovery.py           # ViT EAP-IG
circuit_discovery_swin.py      # Swin EAP-IG + EAP (verbatim from Swin notebook cells 8, 9)
circuit_discovery_gemma.py     # Gemma EAP-IG (verbatim from Gemma notebook cell 13)
gemma_utils.py                 # Prompts, answer matching, zero-shot eval, confused-class
                               # (verbatim from Gemma notebook cells 8, 9)
training.py                    # ViT/Swin/CBIS train_and_evaluate
training_swin.py               # Swin-specific training (verbatim from Swin notebook cell 10)
training_gemma.py              # Gemma generative training (verbatim from Gemma notebook cell 15)
run_cft.py                     # Unified CLI: run_vit() / run_swin() / run_gemma() dispatch
prep_cbis.py                   # Kaggle CBIS-DDSM → VTAB-format converter
setup_vtab.sh                  # Download VTAB-1K
setup_cbis.sh                  # Baseline / data-setup helper from the cbis-ddsm repo
requirements.txt
```

Files prefixed `_swin` / `_gemma` were ported as verbatim as possible from the
original Jupyter notebooks; the dispatcher in `run_cft.py` (functions `run_swin`
and `run_gemma`) is the only substantial new orchestration code. Per-task best
hyperparameters live in `config.py` and were copied from the respective Optuna /
manual-tuning sources — no grid-search code is included.

## Expected behavior summary

- **ViT + VTAB-1K**: matches the reported numbers in the CFT-main repo —
  19 tasks, mean ≈ 75 % at a 17 % parameter budget.
- **Swin (v2) + VTAB-1K**: matches the numbers from `Swin_vtab1k_CFT.ipynb`.
- **ViT + CBIS-DDSM**: single binary classification, best per-task config
  `lr=3e-4, wd=5e-2, smoothing=0.2, budget=12 %`.
- **Gemma + CUB-200**: zero-shot pass first to compute confused-class pairs,
  then EAP-IG with those pairs, then generative training.
