import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm

from config import CLASSES, TrainingConfig
from data import create_dataloaders
from models import build_model, count_trainable_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train chest X-ray image classification models.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", choices=["custom_cnn", "densenet121"], default="custom_cnn")
    parser.add_argument("--strategy", choices=["finetune", "feature_extractor"], default="finetune")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    losses = []
    predictions = []
    targets = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(labels.detach().cpu().tolist())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(targets, predictions)),
    }


def main():
    args = parse_args()
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 5e-4 if args.model == "custom_cnn" else 1e-4

    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model=args.model,
        strategy=args.strategy,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = create_dataloaders(
        config.data_dir,
        config.batch_size,
        config.image_size,
        config.num_workers,
    )

    model = build_model(config.model, config.strategy, num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03 if config.model == "custom_cnn" else 0.0)
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
    best_accuracy = 0.0
    best_path = config.output_dir / "best_model.pt"
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_metrics["accuracy"])
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": config.model,
                    "strategy": config.strategy,
                    "classes": CLASSES,
                    "best_val_accuracy": best_accuracy,
                },
                best_path,
            )

    (config.output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Saved checkpoint: {best_path}")


if __name__ == "__main__":
    main()
