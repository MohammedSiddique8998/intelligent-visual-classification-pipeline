from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import CLASSES


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224):
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(image_size, scale=(0.90, 1.0), ratio=(0.97, 1.03)),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def validate_dataset_structure(data_dir: Path) -> None:
    missing = []
    for split in ("train", "val", "test"):
        for class_name in CLASSES:
            class_dir = data_dir / split / class_name
            if not class_dir.exists():
                missing.append(str(class_dir))
    if missing:
        formatted = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Dataset folders missing:\n{formatted}")


def build_dataloaders(data_dir: Path, batch_size: int, image_size: int, num_workers: int = 2):
    validate_dataset_structure(data_dir)
    train_transform, eval_transform = build_transforms(image_size)

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=eval_transform)

    if tuple(train_dataset.classes) != CLASSES:
        raise ValueError(f"Expected classes {CLASSES}, found {train_dataset.classes}")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def class_counts(dataset: datasets.ImageFolder) -> dict[str, int]:
    counts = {class_name: 0 for class_name in dataset.classes}
    for _, label in dataset.samples:
        counts[dataset.classes[label]] += 1
    return counts


def make_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = np.array([class_counts(dataset)[class_name] for class_name in dataset.classes], dtype=np.float32)
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


create_dataloaders = build_dataloaders
