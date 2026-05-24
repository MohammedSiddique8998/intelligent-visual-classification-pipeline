import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from config import CLASSES
from data import build_dataloaders
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained image-classification checkpoint.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", choices=["custom_cnn", "densenet121"], default="custom_cnn")
    parser.add_argument("--strategy", choices=["finetune", "feature_extractor"], default="finetune")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path, normalised: bool = False) -> None:
    plt.figure(figsize=(7, 6))
    fmt = ".2f" if normalised else "d"
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalised Confusion Matrix" if normalised else "Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def evaluate_model(model, test_loader, device):
    model.eval()

    y_true = []
    y_pred = []
    y_score = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_score.extend(probabilities.cpu().tolist())
            y_true.extend(labels.tolist())

    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)
    normalised_matrix = confusion_matrix(y_true, y_pred, normalize="true")
    roc_auc = None
    try:
        y_true_binary = label_binarize(y_true, classes=list(range(len(CLASSES))))
        roc_auc = float(roc_auc_score(y_true_binary, np.array(y_score), average="macro", multi_class="ovr"))
    except ValueError:
        roc_auc = None

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_roc_auc_ovr": roc_auc,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "normalised_confusion_matrix": normalised_matrix.tolist(),
    }
    return metrics


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = build_dataloaders(args.data_dir, args.batch_size, args.image_size, args.num_workers)
    model = build_model(args.model, args.strategy, num_classes=len(CLASSES)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_model(model, test_loader, device)

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), args.output_dir / "confusion_matrix.png")
    plot_confusion_matrix(
        np.array(metrics["normalised_confusion_matrix"]),
        args.output_dir / "normalised_confusion_matrix.png",
        normalised=True,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
