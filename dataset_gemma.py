"""
Gemma dataset loaders — extracted verbatim from CFT_Gemma3_4B_IT_CUB200.ipynb cell 5.

Used only for Gemma backbone (CUB-200 via PIL access for the VLM processor).
"""
import os
import re
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class _PathLabelDataset(Dataset):
    """Generic dataset from list of (path, label) tuples."""
    def __init__(self, samples, transform, num_classes):
        self.samples = samples
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img) if self.transform else img
        return tensor, label, path


class GPUCachedDataset(Dataset):
    """Pre-load all images as tensors on GPU for fast training."""
    def __init__(self, base_dataset, device):
        self.device = device
        self.num_classes = base_dataset.num_classes
        self.img_paths = [s[0] for s in base_dataset.samples]
        print(f"  Caching {len(base_dataset)} images to {device}...", end=" ", flush=True)

        self.images = []
        self.labels = []
        loader = DataLoader(base_dataset, batch_size=64, num_workers=4,
                            pin_memory=True, shuffle=False,
                            collate_fn=lambda batch: (
                                torch.stack([b[0] for b in batch]),
                                torch.tensor([b[1] for b in batch]),
                                [b[2] for b in batch],
                            ))
        for imgs, labs, _ in loader:
            self.images.append(imgs.to(device))
            self.labels.append(labs.to(device))

        self.images = torch.cat(self.images, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        print(f"Done. Shape: {self.images.shape}")

    def __len__(self):
        return len(self.labels)

    def get_samples_by_class(self):
        by_class = defaultdict(list)
        for i in range(len(self.labels)):
            by_class[self.labels[i].item()].append(i)
        return dict(by_class)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def get_pil_image(self, idx):
        return Image.open(self.img_paths[idx]).convert("RGB")

    def get_label(self, idx):
        return self.labels[idx].item()


def get_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _load_cub200(data_dir, config):
    cub_root = os.path.join(data_dir, "cub200", "CUB_200_2011")
    images = pd.read_csv(os.path.join(cub_root, "images.txt"),
                         sep=" ", names=["img_id", "filepath"])
    labels = pd.read_csv(os.path.join(cub_root, "image_class_labels.txt"),
                         sep=" ", names=["img_id", "target"])
    split = pd.read_csv(os.path.join(cub_root, "train_test_split.txt"),
                        sep=" ", names=["img_id", "is_training_img"])
    data = images.merge(labels, on="img_id").merge(split, on="img_id")
    num_classes = 200
    tfm = get_transforms(config["image_size"])

    train_samples = [
        (os.path.join(cub_root, "images", row.filepath), row.target - 1)
        for _, row in data[data.is_training_img == 1].iterrows()
    ]
    test_samples = [
        (os.path.join(cub_root, "images", row.filepath), row.target - 1)
        for _, row in data[data.is_training_img == 0].iterrows()
    ]

    train_ds = _PathLabelDataset(train_samples, tfm, num_classes)
    test_ds  = _PathLabelDataset(test_samples, tfm, num_classes)
    return train_ds, test_ds, num_classes


def load_vtab_task(task_name, config):
    """Load train and test datasets for a task."""
    train_ds, test_ds, num_classes = _load_cub200(config["data_dir"], config)

    print(f"  Task: {task_name} | Train: {len(train_ds)} | Test: {len(test_ds)} | Classes: {num_classes}")

    if config["use_gpu_cache"] and torch.cuda.is_available():
        train_ds = GPUCachedDataset(train_ds, device)
        test_ds  = GPUCachedDataset(test_ds, device)

    return train_ds, test_ds, num_classes




# =============================================================================
# CUB-200 class names loader (extracted from gemma_cell_3.py lines 9-17)
# =============================================================================
def load_cub_class_names(data_dir):
    """Read class names from CUB classes.txt. Falls back to placeholders."""
    cub_classes_path = os.path.join(data_dir, "cub200", "CUB_200_2011", "classes.txt")
    if os.path.exists(cub_classes_path):
        with open(cub_classes_path) as f:
            return [
                re.sub(r'^\d+\.', '',
                       line.strip().split(" ", 1)[1].replace("_", " ")).strip()
                for line in f
            ]
    return [f"class_{i}" for i in range(200)]
