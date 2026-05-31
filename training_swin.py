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
