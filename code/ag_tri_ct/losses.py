from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional


def soft_dice_loss(logits: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    target_float = target.to(dtype=probabilities.dtype)
    intersection = (probabilities * target_float).sum(dim=(-3, -2, -1))
    denominator = probabilities.sum(dim=(-3, -2, -1)) + target_float.sum(dim=(-3, -2, -1))
    return (1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def fragment_class_loss(logits: Tensor, target: Tensor) -> Tensor:
    return functional.cross_entropy(logits, target)


def signed_distance_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return functional.smooth_l1_loss(prediction, target)


def retrieval_contrastive_loss(
    query: Tensor, positive: Tensor, temperature: float = 0.07
) -> Tensor:
    query_normalized = functional.normalize(query, dim=-1)
    positive_normalized = functional.normalize(positive, dim=-1)
    logits = query_normalized @ positive_normalized.t() / temperature
    labels = torch.arange(query.shape[0], device=query.device)
    return functional.cross_entropy(logits, labels)


def tracing_loss(probabilities: Tensor, responses: Tensor, mask: Tensor) -> Tensor:
    values = functional.binary_cross_entropy(probabilities, responses.float(), reduction="none")
    weighted = values * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


@dataclass(frozen=True)
class JointLoss:
    total: Tensor
    geometry: Tensor
    text: Tensor
    tracing: Tensor
    grounding: Tensor
    fragment: Tensor


def combine_losses(
    geometry: Tensor,
    text: Tensor,
    tracing: Tensor,
    grounding: Tensor,
    fragment: Tensor,
    geometry_weight: float = 0.5,
    text_weight: float = 0.3,
    tracing_weight: float = 0.2,
    fragment_weight: float = 0.3,
) -> JointLoss:
    segmentation = geometry + fragment_weight * fragment
    total = (
        geometry_weight * segmentation + text_weight * text + tracing_weight * tracing + grounding
    )
    return JointLoss(total, geometry, text, tracing, grounding, fragment)
