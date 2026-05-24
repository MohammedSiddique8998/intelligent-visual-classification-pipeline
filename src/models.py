import torch
from torch import nn
from torchvision import models

from config import CLASSES


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class CustomChestXrayCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        channels = (3, 32, 64, 128, 256, 512)
        dropouts = (0.05, 0.08, 0.12, 0.18, 0.22)
        self.features = nn.Sequential(
            *[
                ConvBlock(channels[index], channels[index + 1], dropouts[index])
                for index in range(len(channels) - 1)
            ]
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(model_name: str, strategy: str = "finetune", num_classes: int = len(CLASSES)):
    model_name = model_name.lower()
    strategy = strategy.lower()

    if model_name == "custom_cnn":
        return CustomChestXrayCNN(num_classes=num_classes)

    if model_name != "densenet121":
        raise ValueError("model_name must be 'custom_cnn' or 'densenet121'")

    weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    if strategy == "feature_extractor":
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    elif strategy != "finetune":
        raise ValueError("strategy must be 'finetune' or 'feature_extractor'")

    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
