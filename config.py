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
    "cft_discovery_pct":  10,     # % of train data for circuit discovery
    "cft_param_budget":   15,     # % of total backbone params to unfreeze
    "cft_ig_steps":       8,     # Integrated gradient steps
    "cft_batch_size":     32,     # Batch size for EAP-IG
    # -- Output --
    "save_dir":         os.path.join(os.path.dirname(__file__), "results"),
    "seed":             227,
}

# -- Per-task configs for CFT (tune per task as needed) --
# Best CFT hyperparameters per task
CFT_TASK_CONFIGS = {
    "caltech101":             {"lr": 3.0000e-04, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.1000, "stop_after": 16},
    "cifar":                  {"lr": 3.0000e-04, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 128, "cft_budget": 10, "dropout": 0.1500, "stop_after": 36},
    "clevr_count":            {"lr": 3.0000e-04, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.1500, "stop_after": 25},
    "clevr_dist":             {"lr": 2.9579e-04, "wd": 1.2181e-03, "label_smoothing": 0.3, "batch_size": 32,  "cft_budget": 17, "dropout": 0.1431, "stop_after": 11},
    "dmlab":                  {"lr": 2.4249e-04, "wd": 6.2891e-02, "label_smoothing": 0.0, "batch_size": 32,  "cft_budget": 17, "dropout": 0.0116, "stop_after": 17},
    "dtd":                    {"lr": 2.3172e-04, "wd": 7.1145e-02, "label_smoothing": 0.0, "batch_size": 32,  "cft_budget": 12, "dropout": 0.0637, "stop_after": 17},
    "eurosat":                {"lr": 1.0000e-03, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.2000, "stop_after": 19},
    "kitti":                  {"lr": 5.0000e-04, "wd": 3.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.2500, "stop_after": 26},
    "oxford_iiit_pet":        {"lr": 1.0000e-03, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.0000, "stop_after": 14},
    "patch_camelyon":         {"lr": 1.0000e-03, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.1000, "stop_after": 8},
    "resisc45":               {"lr": 1.0000e-03, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.1000, "stop_after": 22},
    "smallnorb_azi":          {"lr": 2.8507e-04, "wd": 1.9234e-02, "label_smoothing": 0.0, "batch_size": 32,  "cft_budget": 12, "dropout": 0.1082, "stop_after": 15},
    "smallnorb_ele":          {"lr": 3.0000e-04, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.3000, "stop_after": 46},
    "sun397":                 {"lr": 5.0000e-04, "wd": 3.0000e-02, "label_smoothing": 0.1, "batch_size": 128, "cft_budget": 10, "dropout": 0.1000, "stop_after": 32},
    "svhn":                   {"lr": 3.8384e-04, "wd": 4.1056e-02, "label_smoothing": 0.3, "batch_size": 16,  "cft_budget": 17, "dropout": 0.0604, "stop_after": 41},
    "oxford_flowers102":      {"lr": 1.0000e-03, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.0000, "stop_after": 8},
    "diabetic_retinopathy":   {"lr": 5.0000e-04, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 128, "cft_budget": 10, "dropout": 0.3000, "stop_after": 15},
    "dsprites_loc":           {"lr": 1.0000e-03, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.1500, "stop_after": 46},
    "dsprites_ori":           {"lr": 3.0000e-04, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.1500, "stop_after": 68},
    "cbis_ddsm":              {"lr": 3.0000e-04, "wd": 1.0000e-02, "label_smoothing": 0.1, "batch_size": 64,  "cft_budget": 17, "dropout": 0.1000, "stop_after": 15},
    "cbis_ddsm_b12":          {"lr": 3.0000e-04, "wd": 5.0000e-02, "label_smoothing": 0.2, "batch_size": 32,  "cft_budget": 12, "dropout": 0.3000, "stop_after": 5},
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
# Prefix all symbols with SWIN_ to avoid colliding with ViT config above.
# =============================================================================
# =============================================================================
# =============================================================================
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
    "save_dir":         "/cft_benchmark/results",
    "seed":             292,
}

SWIN_TASK_CONFIGS = {
    "caltech101":           {"lr": 5e-4, "wd": 0.01, "label_smoothing": 0.1,  "batch_size": 64, "cft_budget": 17, "dropout": 0.00, "stop_after": 50},
    "cifar":                {"lr": 5e-4, "wd": 0.01, "label_smoothing": 0.1,  "batch_size": 64, "cft_budget": 17, "dropout": 0.05, "stop_after": 50},
    "dtd":                  {"lr": 5e-4, "wd": 0.01, "label_smoothing": 0.1,  "batch_size": 64, "cft_budget": 17, "dropout": 0.00, "stop_after": 50},
    "oxford_flowers102":    {"lr": 5e-4, "wd": 0.01, "label_smoothing": 0.1,  "batch_size": 64, "cft_budget": 17, "dropout": 0.00, "stop_after": 50},
    "oxford_iiit_pet":      {"lr": 5e-4, "wd": 0.01, "label_smoothing": 0.1,  "batch_size": 64, "cft_budget": 17, "dropout": 0.00, "stop_after": 50},
    "sun397":               {"lr": 5e-4, "wd": 0.01, "label_smoothing": 0.1,  "batch_size": 64, "cft_budget": 17, "dropout": 0.05, "stop_after": 50},
    "svhn":                 {"lr": 5e-4, "wd": 0.01, "label_smoothing": 0.1,  "batch_size": 64, "cft_budget": 17, "dropout": 0.00, "stop_after": 50},
    "diabetic_retinopathy": {"lr": 3e-4, "wd": 0.01, "label_smoothing": 0.15, "batch_size": 64, "cft_budget": 17, "dropout": 0.10, "stop_after": 50},
    "eurosat":              {"lr": 5e-4, "wd": 0.01, "label_smoothing": 0.1,  "batch_size": 64, "cft_budget": 17, "dropout": 0.00, "stop_after": 50},
    "resisc45":             {"lr": 5e-4, "wd": 0.01, "label_smoothing": 0.1,  "batch_size": 64, "cft_budget": 20, "dropout": 0.00, "stop_after": 50},
    "dmlab":                {"lr": 3e-4, "wd": 0.01, "label_smoothing": 0.15, "batch_size": 64, "cft_budget": 17, "dropout": 0.10, "stop_after": 50},
    "dsprites_loc":         {"lr": 5e-4, "wd": 0.01, "label_smoothing": 0.15, "batch_size": 64, "cft_budget": 20, "dropout": 0.05, "stop_after": 50},
    "dsprites_ori":         {"lr": 5e-4, "wd": 0.01, "label_smoothing": 0.1,  "batch_size": 64, "cft_budget": 17, "dropout": 0.05, "stop_after": 100},
    "kitti":                {"lr": 5e-4, "wd": 0.01, "label_smoothing": 0.1,  "batch_size": 64, "cft_budget": 17, "dropout": 0.00, "stop_after": 50},
    "patch_camelyon":       {"lr": 5e-5, "wd": 0.01, "label_smoothing": 0.2,  "batch_size": 64, "cft_budget": 20, "dropout": 0.15, "stop_after": 50},
    "clevr_count":          {"lr": 1e-4, "wd": 0.01, "label_smoothing": 0.3,  "batch_size": 64, "cft_budget": 25, "dropout": 0.15, "stop_after": 50},
    "clevr_dist":           {"lr": 1e-4, "wd": 0.01, "label_smoothing": 0.3,  "batch_size": 64, "cft_budget": 25, "dropout": 0.15, "stop_after": 50},
    "smallnorb_azi":        {"lr": 1e-4, "wd": 0.01, "label_smoothing": 0.4,  "batch_size": 64, "cft_budget": 25, "dropout": 0.20, "stop_after": 50},
    "smallnorb_ele":        {"lr": 1e-4, "wd": 0.01, "label_smoothing": 0.4,  "batch_size": 64, "cft_budget": 25, "dropout": 0.20, "stop_after": 50},
}
# =============================================================================
# =============================================================================
# GEMMA BACKBONE CONFIG
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
# =============================================================================
GEMMA_CONFIG = {
    # ── Model ──
    "model_name":       "google/gemma-3-4b-it",
    "image_size":       256,
    "patch_size":       4,

    # ── Data ──
    "data_dir":         "/cft_benchmark/fgvc",
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
    "save_dir":         "/cft_benchmark/results",
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
        return SWIN_TASK_CONFIGS
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
