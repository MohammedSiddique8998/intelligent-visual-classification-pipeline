import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from tqdm import tqdm

from config import CLASSES, load_config
from data import build_dataloaders, class_counts, make_class_weights
from models import build_model, count_trainable_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train chest X-ray image classification models.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", choices=["custom_cnn", "densenet121"], default="custom_cnn")
    parser.add_argument("--strategy", choices=["finetune", "feature_extractor"], default="finetune")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-class-weights", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    predictions = []
    targets = []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    return calculate_epoch_metrics(losses, targets, predictions)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    return calculate_epoch_metrics(losses, targets, predictions)


def calculate_epoch_metrics(losses, targets, predictions):
    if not targets:
        raise ValueError("No samples were processed in this epoch.")

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


def plot_results(history: list[dict], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_macro_f1"] for row in history], label="Train macro F1")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="Validation macro F1")
    axes[1].plot(epochs, [row["val_balanced_accuracy"] for row in history], label="Validation balanced accuracy")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    config = load_config(parse_args())

    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = build_dataloaders(
        config.data_dir,
        config.batch_size,
        config.image_size,
        config.num_workers,
    )

    model = build_model(config.model, config.strategy, num_classes=len(CLASSES)).to(device)
    class_weight_tensor = make_class_weights(train_loader.dataset).to(device) if config.use_class_weights else None
    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
        label_smoothing=0.03 if config.model == "custom_cnn" else 0.0,
    )
    optimizer_cls = torch.optim.AdamW if config.model == "custom_cnn" else torch.optim.Adam
    optimizer = optimizer_cls(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.4,
        patience=3,
    )

    history = []
    best_score = 0.0
    best_path = config.output_dir / "best_model.pt"
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Train class counts: {class_counts(train_loader.dataset)}")

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": config.model,
                    "strategy": config.strategy,
                    "classes": CLASSES,
                    "best_val_macro_f1": best_score,
                    "config": {
                        "seed": config.seed,
                        "batch_size": config.batch_size,
                        "image_size": config.image_size,
                        "epochs": config.epochs,
                        "learning_rate": config.learning_rate,
                        "weight_decay": config.weight_decay,
                        "use_class_weights": config.use_class_weights,
                    },
                },
                best_path,
            )

    (config.output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_results(history, config.output_dir / "training_curves.png")
    print(f"Best validation macro F1: {best_score:.4f}")
    print(f"Saved checkpoint: {best_path}")


if __name__ == "__main__":
    main()
