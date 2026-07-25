"""
PyTorch Dataset/DataLoader for the makeup classification task.

Reads data/manifest.csv (produced by preprocess.py) and serves the already
face-cropped/aligned images from 'processed_path', split by the 'split' column.
"""

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MANIFEST_PATH = "data/manifest.csv"
IMAGE_SIZE = 224

# ImageNet normalization stats: required since we're using an ImageNet-pretrained backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str):
    """
    Augmentation policy:
      - Geometric augmentation (flip, small rotation) is safe.
      - Deliberately NO hue/saturation jitter: makeup is a color signal
        (lipstick, blush, eyeshadow); aggressive color augmentation would
        blur or destroy the exact thing the model needs to learn.
      - Mild brightness/contrast jitter only, to add some lighting robustness
        without washing out color-based makeup cues.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.0, hue=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:  # val/test -- no augmentation, deterministic eval
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class MakeupDataset(Dataset):
    def __init__(self, manifest_path: str = MANIFEST_PATH, split: str = "train"):
        df = pd.read_csv(manifest_path)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got: {split}")

        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found for split='{split}' in {manifest_path}")

        self.transform = get_transforms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["processed_path"]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return img, label

    def class_counts(self):
        """Returns {0: count_no_makeup, 1: count_makeup} for this split."""
        return self.df["label"].value_counts().to_dict()


def get_dataloaders(manifest_path: str = MANIFEST_PATH, batch_size: int = 32, num_workers: int = 4):
    train_ds = MakeupDataset(manifest_path, split="train")
    val_ds = MakeupDataset(manifest_path, split="val")
    test_ds = MakeupDataset(manifest_path, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    train_ds = train_loader.dataset
    print("Train class counts:", train_ds.class_counts())
    print("Val class counts:", val_loader.dataset.class_counts())
    print("Test class counts:", test_loader.dataset.class_counts())

    # Sanity check: pull one batch and print shapes
    imgs, labels = next(iter(train_loader))
    print("Batch image shape:", imgs.shape)   # expect [batch_size, 3, 224, 224]
    print("Batch label shape:", labels.shape) # expect [batch_size]