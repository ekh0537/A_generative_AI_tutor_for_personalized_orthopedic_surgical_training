from dataclasses import dataclass

import numpy as np
from torch import Tensor


def dice_score(prediction: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    prediction_bool = prediction.bool()
    target_bool = target.bool()
    intersection = (prediction_bool & target_bool).sum(dim=(-3, -2, -1)).float()
    total = prediction_bool.sum(dim=(-3, -2, -1)) + target_bool.sum(dim=(-3, -2, -1))
    return ((2.0 * intersection + epsilon) / (total.float() + epsilon)).mean()


def binary_auc(scores: Tensor, labels: Tensor) -> float:
    scores_array = scores.detach().cpu().numpy().astype(np.float64)
    labels_array = labels.detach().cpu().numpy().astype(np.int64)
    positives = scores_array[labels_array == 1]
    negatives = scores_array[labels_array == 0]
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("AUC requires both classes")
    comparisons = (positives[:, None] > negatives[None, :]).mean()
    ties = (positives[:, None] == negatives[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def hallucination_rate(supported: Tensor) -> float:
    if supported.numel() == 0:
        raise ValueError("support vector cannot be empty")
    return float(1.0 - supported.float().mean().item())


def grounding_score(bound_spans: int, total_spans: int) -> float:
    if total_spans <= 0:
        raise ValueError("total spans must be positive")
    return bound_spans / total_spans


@dataclass(frozen=True)
class CompositeMetrics:
    segmentation_dice: float
    text_accuracy: float
    tracing_auc: float
    hallucination_rate: float

    def composite(self, weights: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.1)) -> float:
        if abs(sum(weights) - 1.0) > 1e-8:
            raise ValueError("weights must sum to one")
        return (
            weights[0] * self.segmentation_dice
            + weights[1] * self.text_accuracy
            + weights[2] * self.tracing_auc
            + weights[3] * (1.0 - self.hallucination_rate)
        )
