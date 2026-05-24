from dataclasses import dataclass
from pathlib import Path


CLASSES = ("Normal", "Opacity", "Pneumonia")


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path
    output_dir: Path
    model: str = "custom_cnn"
    strategy: str = "finetune"
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 2
    image_size: int = 224
