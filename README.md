# CFT - Circuit Fine-Tuning

Unified command-line implementation of **Circuit Fine-Tuning (CFT)** for three
backbones - **ViT, Swin (v2), and Gemma-3-4B** - on three datasets:
**VTAB-1K, CBIS-DDSM, and CUB-200**.

CFT is a parameter-efficient fine-tuning (PEFT) method that uses
**EAP-IG** (Edge Attribution Patching with Integrated Gradients,
[Hanna et al., COLM 2024](https://arxiv.org/abs/2403.17806))
to identify a small subset of attention heads and MLP blocks that are most
important for a downstream task. Only those discovered "circuit" components are
unfrozen during fine-tuning, giving-finetune-quality accuracy at a fraction
of the compute budget against PEFT methods.

## What CFT does, end-to-end

For each (backbone, task) pair the pipeline is:

1. **Load** the pretrained backbone with a near-zero-initialized classifier head.
2. **Discover circuits** - run EAP-IG on a slice of training data. The score of a
   node (attention head / MLP) is its average gradient-times-difference between
   *clean* and *corrupted* activations along the integrated-gradient path from
   corrupted to clean inputs. Higher score ⇒ more task-relevant.
3. **Normalize scores** - MLP scores are divided by `mlp_params / head_params`
   so heads and MLPs are compared per equivalent parameter chunk.
4. **Select nodes** under a parameter budget (`--budget`, per-task default from
   the task config tables).
5. **Rebuild the model** with a Kaiming-initialized head, mask gradients to the
   selected nodes, and train for the per-task epoch count.

Corrupted inputs are produced by patch-shuffling, gaussian noise,
channel-shuffling, intensity inversion, or cutout (`--corruption`).

## Quick start

```bash
# 0. install deps
pip install -r requirements.txt

# 1. download VTAB-1K (about 1 GB, one-time)
bash setup_vtab.sh                  # downloads to ./data/vtab-1k

# 2. run CFT on ViT for all 19 VTAB tasks (uses the best per-task HPs)
python run_cft.py --backbone vit --dataset vtab

# 3. ...or on Swin
python run_cft.py --backbone swin --dataset vtab

# 4. ...or just a few tasks
python run_cft.py --backbone vit --dataset vtab --tasks cifar dtd svhn
```

## Backbones supported

| `--backbone` | HuggingFace model id                       | Notes                                          |
| ------------ | ------------------------------------------ | ---------------------------------------------- |
| `vit`        | `google/vit-base-patch16-224-in21k`        | Default. 224 × 224 input.                      |
| `swin`       | `microsoft/swinv2-base-patch4-window8-256` | 256 × 256 input.                               |
| `gemma`      | `google/gemma-3-4b-it`                     | VLM, generative fine-tuning. `--dataset cub200`. |

`google/gemma-3-4b-it` is gated: accept the license on its model page and run
`huggingface-cli login` (or set `HF_TOKEN`) before the first run.

## CLI flags

```
--backbone {vit, swin, gemma}     # default vit
--dataset  {vtab, cbis, cub200}   # default vtab
--tasks    <name1 name2 ...>      # subset; default = all tasks for that dataset
--data-dir <path>                 # override default data location
--budget   <float>                # CFT param budget % (default per-task)
--epochs   <int>                  # override num_epochs
--stop-after-epoch <int>          # early-stop cap
--batch-size <int>
--cft-batch-size <int>            # batch size for EAP-IG discovery
--ig-steps <int>                  # integrated gradient steps
--discovery-pct <float>           # % of training data used for discovery
--corruption {patch_shuffle, gaussian, channel_shuffle, intensity_invert, cutout, multi, multi_med}
--method {eap-ig, eap}            # eap-ig uses the IG path; eap is a single gradient
--method-tag <str>                # label saved in the results JSON
--lr / --wd / --label-smoothing / --dropout    # per-task overrides
--ddp                             # DistributedDataParallel (launch with torchrun)
```

Run `python run_cft.py --help` for the full list.
