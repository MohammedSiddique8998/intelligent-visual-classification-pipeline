from dataclasses import dataclass
from pathlib import Path


CLASSES = ("Normal", "Opacity", "Pneumonia")
DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path
    output_dir: Path
    model: str = "custom_cnn"
    strategy: str = "finetune"
    epochs: int = 30
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    seed: int = DEFAULT_SEED
    num_workers: int = 2
    image_size: int = DEFAULT_IMAGE_SIZE
    use_class_weights: bool = False


def load_config(args) -> TrainingConfig:
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 5e-4 if args.model == "custom_cnn" else 1e-4

    return TrainingConfig(
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
        image_size=args.image_size,
        use_class_weights=args.use_class_weights,
    )
