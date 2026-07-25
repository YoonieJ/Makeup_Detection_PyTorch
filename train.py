"""
Training script for the makeup/no-makeup binary classifier.

Fine-tunes an ImageNet-pretrained ResNet18 on data/manifest.csv (via
dataset.py), using a class-weighted BCE loss to correct for the
makeup/no-makeup imbalance in the splits, and checkpoints the model with the
best validation F1.
"""

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18

from dataset import get_dataloaders

CHECKPOINT_DIR = Path("checkpoints")  # gitignored; recreated by main() each run


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    """ImageNet-pretrained ResNet18 with the 1000-way head replaced by a
    single logit, since this is binary classification (BCEWithLogitsLoss
    expects one raw score per sample, not two)."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def get_device() -> torch.device:
    # Apple Silicon target per README (MPS); falls back to CUDA/CPU so the
    # script also runs elsewhere.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pos_weight_from_counts(class_counts: dict) -> torch.Tensor:
    """BCEWithLogitsLoss pos_weight for the (1=makeup, 0=no_makeup) label
    convention used throughout dataset.py/preprocess.py.

    pos_weight = n_negative / n_positive. This is a ratio, not an "is the
    positive class rare" check: on this dataset makeup (1) is the *majority*
    class (~768 vs 322 in train), so pos_weight comes out < 1, which shrinks
    the positive term's contribution relative to the fixed-weight negative
    term -- i.e. it corrects for imbalance in whichever direction it runs,
    not just when the positive class happens to be the minority one.
    """
    n_pos = class_counts.get(1, 0)
    n_neg = class_counts.get(0, 0)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, criterion) -> dict:
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        # model(imgs) is [batch, 1] (one logit per sample); squeeze to [batch]
        # to match the label shape MakeupDataset produces (dataset.py:66).
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)

        probs = torch.sigmoid(logits)
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    # average="binary" scores against label 1 (makeup) as the positive class;
    # zero_division=0 avoids a warning/crash on a split with no predicted
    # positives, which can happen for a few epochs on a small val set.
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Raised if a batch/split ends up with only one class present.
        auc = float("nan")
    accuracy = sum(p == l for p, l in zip(preds, all_labels)) / len(all_labels)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def train_one_epoch(model, loader, device, criterion, optimizer) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def format_metrics(metrics: dict) -> str:
    return (
        f"loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} "
        f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the makeup/no-makeup classifier.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience on val F1.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        manifest_path=args.manifest, batch_size=args.batch_size, num_workers=args.num_workers
    )

    pos_weight = pos_weight_from_counts(train_loader.dataset.class_counts()).to(device)
    print(f"Class counts (train): {train_loader.dataset.class_counts()} -> pos_weight={pos_weight.item():.3f}")

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Watches val F1 rather than val loss: loss is computed with the
    # class-weighted criterion above, so it doesn't directly track the
    # precision/recall balance we actually care about on an imbalanced set.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_metrics["f1"])
        elapsed = time.time() - start

        print(f"[epoch {epoch:02d}/{args.epochs}] ({elapsed:.1f}s) "
              f"train_loss={train_loss:.4f} val: {format_metrics(val_metrics)}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            # Snapshot in memory (not just on disk) so the reload below is
            # exact even if training continues to mutate model in place.
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(best_state, CHECKPOINT_DIR / "best_model.pt")
            print(f"  -> new best val F1={best_f1:.4f}, checkpoint saved")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"No val F1 improvement for {args.patience} epochs, stopping early.")
                break

    # Report test performance from the best *validation* checkpoint, not
    # whatever the model happened to be after the last/stopping epoch --
    # avoids reporting an overfit or post-early-stop-drift state.
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device, criterion)
    print(f"\nFinal test metrics (best val checkpoint): {format_metrics(test_metrics)}")


if __name__ == "__main__":
    main()
