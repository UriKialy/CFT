"""
VTAB-1K Fine-Tuning Benchmark — Configuration
"""
import os
import random
import numpy as np
import torch

# =============================================================================
# Main configuration
# =============================================================================
CONFIG = {
    # -- Model --
    "model_name":       "google/vit-base-patch16-224-in21k",
    "image_size":       224,
    "patch_size":       16,

    # -- Data --
    "data_dir":         os.path.join(os.path.dirname(__file__), "..", "data", "vtab-1k"),
    "train_file":       "train800.txt",      # 800 training samples
    "test_file":        "test.txt",         #  test samples
    "use_gpu_cache":    True,                 # Cache tensors on GPU

    # -- Training --
    "batch_size":       256,
    "learning_rate":    1e-4,
    "weight_decay":     0.01,
    "num_epochs":       15,
    "optimizer":        "adamw",
    "scheduler":        "cosine",
    "num_workers":      4,

    # -- CFT (Circuit Fine-Tune) --
    "cft_discovery_pct":  15,     # % of train data for circuit discovery
    "cft_param_budget":   20,     # % of total backbone params to unfreeze
    "cft_ig_steps":       12,     # Integrated gradient steps
    "cft_batch_size":     32,     # Batch size for EAP-IG
    # -- Output --
    "save_dir":         os.path.join(os.path.dirname(__file__), "results"),
    "seed":             42,
}

# -- Per-task configs for CFT (tune per task as needed) --
# Best CFT hyperparameters per task
CFT_TASK_CONFIGS = {
    "caltech101":             {"lr": 3.0000e-04, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.1000, "stop_after": 16, "_source": "disc100_early_ep11", "_best_acc": 96.50},
    "cifar":                  {"lr": 3.0000e-04, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 128, "cft_budget": 10, "dropout": 0.1500, "stop_after": 36, "_source": "disc100", "_best_acc": 74.50},
    "clevr_count":            {"lr": 3.0000e-04, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.1500, "stop_after": 25, "_source": "disc100_early_ep20", "_best_acc": 76.00},
    "clevr_dist":             {"lr": 2.9579e-04, "wd": 1.2181e-03, "label_smoothing": 0.3, "batch_size": 32,  "cft_budget": 17, "dropout": 0.1431, "stop_after": 11, "_source": "optuna_trial_17", "_best_acc": 64.99},
    "dmlab":                  {"lr": 2.4249e-04, "wd": 6.2891e-02, "label_smoothing": 0.0, "batch_size": 32,  "cft_budget": 17, "dropout": 0.0116, "stop_after": 17, "_source": "optuna_trial_6", "_best_acc": 49.49},
    "dtd":                    {"lr": 2.3172e-04, "wd": 7.1145e-02, "label_smoothing": 0.0, "batch_size": 32,  "cft_budget": 12, "dropout": 0.0637, "stop_after": 17, "_source": "optuna_trial_0", "_best_acc": 68.99},
    "eurosat":                {"lr": 1.0000e-03, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.2000, "stop_after": 19, "_source": "disc100_early_ep14", "_best_acc": 95.00},
    "kitti":                  {"lr": 5.0000e-04, "wd": 3.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.2500, "stop_after": 26, "_source": "disc100", "_best_acc": 84.00},
    "oxford_iiit_pet":        {"lr": 1.0000e-03, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.0000, "stop_after": 14, "_source": "disc100_early_ep9", "_best_acc": 93.50},
    "patch_camelyon":         {"lr": 1.0000e-03, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.1000, "stop_after": 8,  "_source": "disc100_early_ep3", "_best_acc": 87.00},
    "resisc45":               {"lr": 1.0000e-03, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.1000, "stop_after": 22, "_source": "disc100_early_ep17", "_best_acc": 86.00},
    "smallnorb_azi":          {"lr": 2.8507e-04, "wd": 1.9234e-02, "label_smoothing": 0.0, "batch_size": 32,  "cft_budget": 12, "dropout": 0.1082, "stop_after": 15, "_source": "optuna_trial_12", "_best_acc": 21.12},
    "smallnorb_ele":          {"lr": 3.0000e-04, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.3000, "stop_after": 46, "_source": "disc100", "_best_acc": 39.50},
    "sun397":                 {"lr": 5.0000e-04, "wd": 3.0000e-02, "label_smoothing": 0.1, "batch_size": 128, "cft_budget": 10, "dropout": 0.1000, "stop_after": 32, "_source": "disc100_early_ep27", "_best_acc": 55.00},
    "svhn":                   {"lr": 3.8384e-04, "wd": 4.1056e-02, "label_smoothing": 0.3, "batch_size": 16,  "cft_budget": 17, "dropout": 0.0604, "stop_after": 41, "_source": "optuna_trial_19", "_best_acc": 88.58},
    "oxford_flowers102":      {"lr": 1.0000e-03, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.0000, "stop_after": 8,  "_source": "disc100_early_ep3", "_best_acc": 98.50},
    "diabetic_retinopathy":   {"lr": 5.0000e-04, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 128, "cft_budget": 10, "dropout": 0.3000, "stop_after": 15, "_source": "disc100", "_best_acc": 76.00},
    "dsprites_loc":           {"lr": 1.0000e-03, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.1500, "stop_after": 46, "_source": "disc100_early_ep41", "_best_acc": 82.00},
    "dsprites_ori":           {"lr": 3.0000e-04, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.1500, "stop_after": 68, "_source": "disc100", "_best_acc": 53.50},
    "cbis_ddsm":              {"lr": 3.0000e-04, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.1000, "stop_after": 15, "_source": "cbis_initial"},
    "cbis_ddsm_b12":          {"lr": 3.0000e-04, "wd": 5.0000e-02, "label_smoothing": 0.2, "batch_size": 32,  "cft_budget": 12, "dropout": 0.3000, "stop_after": 5,  "_source": "cbis_grid_ep5_b12_mid", "_best_acc": 71.88, "_corruption": "cutout"},
}
# =============================================================================
# Task lists
# =============================================================================
VTAB_TASKS = [
    # Natural (7)
    "caltech101", "cifar", "dtd", "oxford_flowers102",
    "oxford_iiit_pet", "sun397", "svhn",
    # Specialized (4)
    "diabetic_retinopathy", "eurosat", "patch_camelyon", "resisc45",
    # Structured (8)
    "clevr_count", "clevr_dist", "dmlab", "dsprites_loc",
    "dsprites_ori", "kitti", "smallnorb_azi", "smallnorb_ele",
]

NATURAL_TASKS = ["cifar", "caltech101", "dtd", "oxford_flowers102",
                 "oxford_iiit_pet", "sun397", "svhn"]
SPECIALIZED_TASKS = ["patch_camelyon", "eurosat", "resisc45", "diabetic_retinopathy"]
STRUCTURED_TASKS = ["clevr_count", "clevr_dist", "dmlab", "kitti",
                    "dsprites_loc", "dsprites_ori", "smallnorb_azi", "smallnorb_ele"]

METHODS = ["cft"]

# Short names for display
TASK_SHORT_NAMES = {
    "cifar": "CIFAR", "caltech101": "Cal101", "dtd": "DTD",
    "oxford_flowers102": "Flwr", "oxford_iiit_pet": "Pets",
    "sun397": "Sun397", "svhn": "SVHN",
    "patch_camelyon": "Camel", "eurosat": "EuroS", "resisc45": "RESI",
    "diabetic_retinopathy": "DRet",
    "clevr_count": "CClnt", "clevr_dist": "CDist", "dmlab": "DMLab",
    "kitti": "KITTI", "dsprites_loc": "DSLoc", "dsprites_ori": "DSOri",
    "smallnorb_azi": "SNAzi", "smallnorb_ele": "SNEle",
}


# =============================================================================
# Device & seed setup
# =============================================================================
def setup_environment(config=None):
    """Set seeds and configure device. Returns the device."""
    if config is None:
        config = CONFIG
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    os.makedirs(config["save_dir"], exist_ok=True)
    return device


# =============================================================================
# =============================================================================
# SWIN BACKBONE CONFIG
# Copied verbatim from Swin_vtab1k_CFT.ipynb cell 5 (CONFIG + best per-task HPs).
# Prefix all symbols with SWIN_ to avoid colliding with ViT config above.
# =============================================================================
# =============================================================================
# =============================================================================
# CELL 2: Configuration
# =============================================================================
SWIN_CONFIG = {
    # ── Model ──
    "model_name":       "microsoft/swinv2-base-patch4-window8-256",
    "image_size":       256,
    "patch_size":       4,

    # ── Data ──
    "data_dir":         "/content/cft_benchmark/vtab-1k",
    "train_file":       "train800.txt",      # 800 training samples
    "test_file":        "test.txt",         #  test samples
    "use_gpu_cache":    True,                 # Cache tensors on GPU

    # ── Training ──
    "batch_size":       512,
    "learning_rate":    1e-4,
    "weight_decay":     0.01,
    "num_epochs":       50,
    "optimizer":        "adamw",
    "scheduler":        "cosine",
    "num_workers":      4,


    # ── CFT (Circuit Fine-Tune) ──
    "cft_discovery_pct":  20,     # % of train data for circuit discovery
    "cft_param_budget":   17,      # % of total backbone params to unfreeze
    "cft_ig_steps":       8,      # Integrated gradient steps
    "cft_batch_size":     32,      # Batch size for EAP-IG

    # ── Output ──
    "save_dir":         "/content/cft_benchmark/results",
    "seed":             42,
}

SWIN_METHOD_BATCH_SIZE = {
    "cft": 64,
}

# ── LEARNING RATES ──

SWIN_CFT_TASK_LRS = {
    "caltech101":           5e-4,
    "cifar":                5e-4,
    "dtd":                  5e-4,
    "oxford_flowers102":    5e-4,
    "oxford_iiit_pet":      5e-4,
    "sun397":               5e-4,
    "svhn":                 5e-4,
    "diabetic_retinopathy": 3e-4,
    "eurosat":              5e-4,
    "resisc45":             5e-4,
    "dmlab":                3e-4,
    "dsprites_loc":         5e-4,
    "dsprites_ori":         5e-4,
    "kitti":                5e-4,
    "patch_camelyon":       5e-5,   
    "clevr_count":          1e-4,  
    "clevr_dist":           1e-4,   
    "smallnorb_azi":        1e-4,  
    "smallnorb_ele":        1e-4,   
}

# ── LABEL SMOOTHING ──
SWIN_CFT_LABEL_SMOOTHING = {
    "clevr_count":          0.3,    
    "clevr_dist":           0.3,    
    "smallnorb_azi":        0.4,   
    "smallnorb_ele":        0.4,    
    "patch_camelyon":       0.2,   
    "dmlab":                0.15,
    "diabetic_retinopathy": 0.15,
    "dsprites_loc":         0.15,
    "dsprites_ori":         0.1,
    # default 0.1 for everything else
}

# ── BUDGET ──
SWIN_CFT_TASK_BUDGETS = {
    "clevr_count":    0.25,  
    "clevr_dist":     0.25,
    "smallnorb_azi":  0.25,   
    "smallnorb_ele":  0.25,
    "dsprites_loc":   0.20,
    "dmlab":          0.17,
    "patch_camelyon":  0.20,  
    "resisc45":       0.20,   
}

# ── EPOCHS ──
SWIN_CFT_TASK_EPOCHS = {
    "caltech101": 50, "cifar": 50, "dtd": 50, "oxford_flowers102": 50,
    "oxford_iiit_pet": 50, "sun397": 50, "svhn": 50,
    "diabetic_retinopathy": 50, "eurosat": 50,
    "resisc45":          50,   
    "patch_camelyon":    50,   
    "clevr_count":       50,  
    "clevr_dist":        50,
    "dmlab":              50,
    "dsprites_loc":       50,
    "dsprites_ori":       100,
    "kitti":              50,
    "smallnorb_azi":     50,   
    "smallnorb_ele":     50,
}

SWIN_CFT_DROPOUT = {
    # Keep what worked
    "caltech101": 0.0, "cifar": 0.05, "dtd": 0.0, "oxford_flowers102": 0.0,
    "oxford_iiit_pet": 0.0, "sun397": 0.05, "svhn": 0.0,
    "diabetic_retinopathy": 0.1, "eurosat": 0.0, "resisc45": 0.0,
    "dmlab": 0.1, "dsprites_loc": 0.05, "dsprites_ori": 0.05, "kitti": 0.0,

    "patch_camelyon": 0.15,   
    "clevr_count":    0.15,    
    "clevr_dist":     0.15,    
    "smallnorb_azi":  0.2,      
    "smallnorb_ele":  0.2,     
}



# =============================================================================
# =============================================================================
# GEMMA BACKBONE CONFIG
# Copied verbatim from CFT_Gemma3_4B_IT_CUB200.ipynb cell 3.
# All Gemma symbols are prefixed GEMMA_ to avoid colliding with ViT config.
# =============================================================================
# =============================================================================
GEMMA_TASKS = ["cub200"]

GEMMA_TASK_CLASS_NAMES = {
    "cub200": None,  
}

GEMMA_TASK_DOMAIN_HINT = {
    "cub200": "a bird species",
}

GEMMA_STRUCTURED_TASK_CONFIG = {}

# =============================================================================
# CELL 2: Configuration
# =============================================================================
GEMMA_CONFIG = {
    # ── Model ──
    "model_name":       "google/gemma-3-4b-it",
    "image_size":       256,
    "patch_size":       4,

    # ── Data ──
    "data_dir":         "/workspace/cft_benchmark/fgvc",
    "use_gpu_cache":    True,

    # ── Training ──
    "batch_size":       32,
    "learning_rate":    1e-4,
    "weight_decay":     0.01,
    "num_epochs":       4,
    "optimizer":        "adamw",
    "scheduler":        "cosine",
    "num_workers":      4,
    "max_new_tokens":   10,
    "batch_size_train": 32,               
    "gradient_accumulation_steps": 4,    

    # ── CFT (Circuit Fine-Tune) ──
    "cft_discovery_pct":  20,
    "cft_param_budget":   17,
    "cft_ig_steps":       8,
    "cft_batch_size":     32,

    # ── Output ──
    "save_dir":         "/workspace/cft_benchmark/results",
    "seed":             42,
}


GEMMA_CFT_DROPOUT = {"cub200": 0.1}

GEMMA_CFT_TASK_LRS = {"cub200": 5e-5}

GEMMA_CFT_TASK_EPOCHS = {"cub200": 4}



def get_backbone_config(backbone):
    """Return the CONFIG dict to use for this backbone.

    Note: the ViT 'CONFIG' dict (defined at top of file) is the default.
    For Swin, returns SWIN_CONFIG. For Gemma, returns GEMMA_CONFIG.
    """
    if backbone == "vit":
        return dict(CONFIG)
    if backbone == "swin":
        return dict(SWIN_CONFIG)
    if backbone == "gemma":
        return dict(GEMMA_CONFIG)
    raise ValueError(f"Unknown backbone: {backbone!r}")


def get_task_configs(backbone):
    """Return the per-task best-HP dict for this backbone.

    For ViT, this is CFT_TASK_CONFIGS (defined at top of file).
    For Swin, returns a per-task dict assembled from SWIN_CFT_TASK_LRS etc.
    For Gemma, returns a single-entry dict for cub200.
    """
    if backbone == "vit":
        return CFT_TASK_CONFIGS
    if backbone == "swin":
        # Assemble per-task dict from the separate Swin dicts (LRs, smoothing,
        # budgets, epochs, dropout). All read from the SWIN_* dicts above.
        per_task = {}
        for t in _SWIN_VTAB_TASKS_UNUSED:
            per_task[t] = {
                "lr":              SWIN_CFT_TASK_LRS.get(t, 5e-4),
                "wd":              0.01,
                "label_smoothing": SWIN_CFT_LABEL_SMOOTHING.get(t, 0.1),
                "batch_size":      SWIN_METHOD_BATCH_SIZE.get("cft", 64),
                "cft_budget":      int(100 * SWIN_CFT_TASK_BUDGETS.get(t, 0.17)),
                "dropout":         SWIN_CFT_DROPOUT.get(t, 0.0),
                "stop_after":      SWIN_CFT_TASK_EPOCHS.get(t, 80),
            }
        return per_task
    if backbone == "gemma":
        return {
            "cub200": {
                "lr":              GEMMA_CFT_TASK_LRS.get("cub200", 5e-5),
                "wd":              GEMMA_CONFIG.get("weight_decay", 0.01),
                "label_smoothing": 0.0,
                "batch_size":      GEMMA_CONFIG.get("batch_size_train", 8),
                "cft_budget":      GEMMA_CONFIG.get("cft_param_budget", 17),
                "dropout":         GEMMA_CFT_DROPOUT.get("cub200", 0.1),
                "stop_after":      GEMMA_CFT_TASK_EPOCHS.get("cub200", 10),
            }
        }
    raise ValueError(f"Unknown backbone: {backbone!r}")
