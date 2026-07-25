"""
Makeup detection - model training.

Loads data/manifest.csv (produced by preprocess/preprocess.py), fine-tunes an
ImageNet-pretrained ResNet18 for binary makeup / no_makeup classification, and
saves the best checkpoint (by validation balanced accuracy) to
train/checkpoints/best.pt. After training, evaluates that checkpoint once on
the held-out test split.

Run order: run preprocess/preprocess.py first to produce data/manifest.csv and
data/processed/, then run this as a script.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

# Config

MANIFEST_PATH = Path("data/manifest.csv")
CHECKPOINT_DIR = Path("train/checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "best.pt"

IMAGE_SIZE = 224            # must match preprocess.py's IMAGE_SIZE
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 6
NUM_WORKERS = 4
RANDOM_SEED = 42

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Standard ImageNet normalization stats, required by the pretrained backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# Dataset

class MakeupDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        self.paths = df["processed_path"].tolist()
        self.labels = df["label"].astype("float32").tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


# Model

def build_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)  # single logit: makeup vs no_makeup
    return model.to(DEVICE)


# Train / eval loop

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, bal_acc, f1


# Main

if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    manifest = pd.read_csv(MANIFEST_PATH)
    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest[manifest["split"] == "val"].reset_index(drop=True)
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)

    train_transform, eval_transform = build_transforms()
    train_loader = DataLoader(MakeupDataset(train_df, train_transform),
                               batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(MakeupDataset(val_df, eval_transform),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(MakeupDataset(test_df, eval_transform),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Makeup is the majority class in this manifest (~69%); down-weight it so
    # the loss doesn't just learn to predict "makeup" for everything.
    num_pos = (train_df["label"] == 1).sum()
    num_neg = (train_df["label"] == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32, device=DEVICE)

    model = build_model()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_bal_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_bal_acc, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_bal_acc, val_f1 = run_epoch(model, val_loader, criterion)
        scheduler.step(val_bal_acc)

        print(f"epoch {epoch:02d} ({time.time() - t0:.1f}s)  "
              f"train_loss={train_loss:.4f} train_bal_acc={train_bal_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_bal_acc={val_bal_acc:.4f} val_f1={val_f1:.4f}")

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_bal_acc": val_bal_acc,
                "val_f1": val_f1,
            }, CHECKPOINT_PATH)
            print(f"  -> saved new best checkpoint (val_bal_acc={val_bal_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    print(f"\nBest val balanced accuracy: {best_val_bal_acc:.4f}  (checkpoint: {CHECKPOINT_PATH})")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_bal_acc, test_f1 = run_epoch(model, test_loader, criterion)
    print(f"Test (best checkpoint, epoch {checkpoint['epoch']}): "
          f"loss={test_loss:.4f} balanced_acc={test_bal_acc:.4f} f1={test_f1:.4f}")
