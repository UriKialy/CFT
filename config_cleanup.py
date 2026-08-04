"""
One-shot patch:
  1. config.py:
     - Strip _source / _best_acc from CFT_TASK_CONFIGS (ViT/VTAB)
     - Replace 6 scattered SWIN_* dicts with a single SWIN_TASK_CONFIGS
     - Remove the Swin assembly branch (uses undefined _SWIN_VTAB_TASKS_UNUSED)
  2. Swin.py:114:  drop CFT_DROPOUT global; read from task config
  3. circuit_discovery.py:403:  drop undefined `metric` from return dict
  4. circuit_discovery_gemma.py:  add missing `import gc`
  5. circuit_discovery_swin.py:340:  replace undefined CFT_TASK_BUDGETS/CONFIG
"""

import re

# ============================================================================
# 1. config.py
# ============================================================================
src = open("config.py").read()

# 1a. Strip _source and _best_acc (and _corruption for cbis_ddsm_b12) from each
#     CFT_TASK_CONFIGS entry.
def strip_metadata(match):
    line = match.group(0)
    # remove all ', "_source": ...' , ', "_best_acc": ...', ', "_corruption": ...'
    line = re.sub(r',\s*"_source":\s*"[^"]*"', '', line)
    line = re.sub(r',\s*"_best_acc":\s*[-\d.]+', '', line)
    line = re.sub(r',\s*"_corruption":\s*"[^"]*"', '', line)
    return line

# Only touch lines inside CFT_TASK_CONFIGS (identified by leading '    "taskname":')
src = re.sub(
    r'^(    "[a-z_0-9]+":\s+\{[^}]*\}),?$',
    strip_metadata,
    src,
    flags=re.MULTILINE,
)

# 1b. Build the SWIN_TASK_CONFIGS block
swin_dict = '''SWIN_TASK_CONFIGS = {
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
'''

# 1c. Replace the whole run of 6 SWIN_* dicts (from SWIN_METHOD_BATCH_SIZE
#     through SWIN_CFT_DROPOUT close-brace) with SWIN_TASK_CONFIGS.
pattern = re.compile(
    r"SWIN_METHOD_BATCH_SIZE = \{.*?^\}\s*\n",
    re.DOTALL | re.MULTILINE,
)
if not pattern.search(src):
    raise SystemExit("could not find SWIN_METHOD_BATCH_SIZE start")

# Find where SWIN_CFT_DROPOUT ends
end_marker = re.search(r"SWIN_CFT_DROPOUT = \{.*?^\}\s*\n", src, re.DOTALL | re.MULTILINE)
if not end_marker:
    raise SystemExit("could not find SWIN_CFT_DROPOUT end")

start_idx = pattern.search(src).start()
end_idx   = end_marker.end()
old_block = src[start_idx:end_idx]
# Sanity: it should contain all 6 dict names
for name in ("SWIN_METHOD_BATCH_SIZE", "SWIN_CFT_TASK_LRS",
             "SWIN_CFT_LABEL_SMOOTHING", "SWIN_CFT_TASK_BUDGETS",
             "SWIN_CFT_TASK_EPOCHS", "SWIN_CFT_DROPOUT"):
    assert name in old_block, f"expected {name} in the SWIN block"

src = src[:start_idx] + swin_dict + src[end_idx:]

# 1d. Replace the swin branch of the factory that uses _SWIN_VTAB_TASKS_UNUSED
old_factory = re.search(
    r"    if backbone == \"swin\":\n(.*?)        return per_task\n",
    src,
    re.DOTALL,
)
if not old_factory:
    raise SystemExit("could not find swin factory branch")
new_factory = '    if backbone == "swin":\n        return SWIN_TASK_CONFIGS\n'
src = src.replace(old_factory.group(0), new_factory)

open("config.py", "w").write(src)
print("config.py OK")


# ============================================================================
# 2. Swin.py: fix CFT_DROPOUT
# ============================================================================
src = open("Swin.py").read()
old = '''    # ── Dropout for unfrozen layers ──
    dropout_rate = CFT_DROPOUT.get(task_name, 0.0)'''
new = '''    # ── Dropout for unfrozen layers (from task config) ──
    dropout_rate = config.get("head_dropout", 0.0)'''
assert old in src, "Swin.py CFT_DROPOUT line not found"
src = src.replace(old, new)
open("Swin.py", "w").write(src)
print("Swin.py OK")


# ============================================================================
# 3. circuit_discovery.py: drop undefined `metric` from return
# ============================================================================
src = open("circuit_discovery.py").read()
old = '''        "method": f"EAP-IG-{metric}",'''
new = '''        "method": "EAP-IG",'''
assert old in src, "circuit_discovery.py metric return line not found"
src = src.replace(old, new)
open("circuit_discovery.py", "w").write(src)
print("circuit_discovery.py OK")


# ============================================================================
# 4. circuit_discovery_gemma.py: add missing `import gc`
# ============================================================================
src = open("circuit_discovery_gemma.py").read()
if "\nimport gc\n" not in src and "import gc\n" not in src.splitlines()[:20]:
    # insert after `import torch` (first occurrence)
    idx = src.find("import torch\n")
    if idx < 0:
        idx = 0
    else:
        idx = src.find("\n", idx) + 1
    src = src[:idx] + "import gc\n" + src[idx:]
    open("circuit_discovery_gemma.py", "w").write(src)
    print("circuit_discovery_gemma.py OK")
else:
    print("circuit_discovery_gemma.py: gc already imported")


# ============================================================================
# 5. circuit_discovery_swin.py: drop undefined CFT_TASK_BUDGETS + CONFIG
# ============================================================================
src = open("circuit_discovery_swin.py").read()
old = '''    budget_pct = CFT_TASK_BUDGETS.get(task_name, CONFIG["cft_param_budget"])
    budget = int(budget_pct / 100 * total_params) if budget_pct > 1 else int(budget_pct * total_params)'''
new = '''    budget = int(target_pct / 100 * total_params) if target_pct > 1 else int(target_pct * total_params)'''
assert old in src, "circuit_discovery_swin.py budget block not found"
src = src.replace(old, new)
open("circuit_discovery_swin.py", "w").write(src)
print("circuit_discovery_swin.py OK")
