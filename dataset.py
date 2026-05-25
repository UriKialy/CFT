"""
VTAB-1K Fine-Tuning Benchmark — Dataset loading and GPU caching
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class VTABDataset(Dataset):
    """Load VTAB-1K task from text file listing (img_path label)."""

    def __init__(self, root, split_file, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        fpath = os.path.join(root, split_file)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Split file not found: {fpath}")

        with open(fpath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_rel = parts[0]
                    label = int(parts[1])
                    self.samples.append((os.path.join(root, img_rel), label))

        labels = [s[1] for s in self.samples]
        self.num_classes = max(labels) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class GPUCachedDataset(Dataset):
    """Pre-load all images as tensors on GPU for fast training."""

    def __init__(self, base_dataset, device):
        self.device = device
        self.num_classes = base_dataset.num_classes
        print(f"  Caching {len(base_dataset)} images to {device}...", end=" ", flush=True)

        self.images = []
        self.labels = []
        loader = DataLoader(base_dataset, batch_size=64, num_workers=4,
                            pin_memory=True, shuffle=False)
        for imgs, labs in loader:
            self.images.append(imgs.to(device))
            self.labels.append(labs.to(device))

        self.images = torch.cat(self.images, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        print(f"Done. Shape: {self.images.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_transforms(image_size):
    """Standard transforms for ViT: resize, center crop, normalize."""
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_vtab_task(task_name, config, device=None):
    """Load train and test datasets for a single VTAB task."""
    root = os.path.join(config["data_dir"], task_name)
    tfm = get_transforms(config["image_size"])

    train_ds = VTABDataset(root, config["train_file"], transform=tfm)
    test_ds  = VTABDataset(root, config["test_file"],  transform=tfm)

    num_classes = train_ds.num_classes
    print(f"  Task: {task_name} | Train: {len(train_ds)} | Test: {len(test_ds)} | Classes: {num_classes}")

    if config["use_gpu_cache"] and torch.cuda.is_available() and device is not None:
        train_ds = GPUCachedDataset(train_ds, device)
        test_ds  = GPUCachedDataset(test_ds, device)

    return train_ds, test_ds, num_classes
