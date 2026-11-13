import torch
from torch import nn

from .experiments import EXPERIMENTS


def binary_cross_entropy_loss() -> nn.Module:
    return nn.BCEWithLogitsLoss()


class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        if not 0.0 <= dice_weight <= 1.0:
            raise ValueError("dice_weight must be between 0 and 1")
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        dimensions = tuple(range(1, targets.ndim))
        intersection = (probabilities * targets).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
        dice = ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        return (1.0 - self.dice_weight) * bce + self.dice_weight * (1.0 - dice)


def loss_for_experiment(experiment: str) -> nn.Module:
    spec = EXPERIMENTS[experiment]
    if spec.dice_weight is None:
        return binary_cross_entropy_loss()
    return BCEDiceLoss(dice_weight=spec.dice_weight)
