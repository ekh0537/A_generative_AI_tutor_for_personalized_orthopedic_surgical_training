from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float


def paired_bootstrap(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    statistic: Callable[[NDArray[np.float64]], float] = np.mean,
    samples: int = 10000,
    seed: int = 2026,
) -> Interval:
    if first.shape != second.shape:
        raise ValueError("paired arrays must have equal shape")
    generator = np.random.default_rng(seed)
    differences = first - second
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selection = generator.integers(0, len(differences), len(differences))
        estimates[index] = statistic(differences[selection])
    return Interval(
        float(statistic(differences)),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def holm_bonferroni(p_values: NDArray[np.float64], alpha: float = 0.05) -> NDArray[np.bool_]:
    order = np.argsort(p_values)
    rejected = np.zeros(len(p_values), dtype=np.bool_)
    for rank, index in enumerate(order):
        threshold = alpha / (len(p_values) - rank)
        if p_values[index] > threshold:
            break
        rejected[index] = True
    return rejected


def benjamini_hochberg(p_values: NDArray[np.float64], alpha: float = 0.05) -> NDArray[np.bool_]:
    order = np.argsort(p_values)
    sorted_values = p_values[order]
    thresholds = alpha * np.arange(1, len(p_values) + 1) / len(p_values)
    qualifying = np.flatnonzero(sorted_values <= thresholds)
    rejected = np.zeros(len(p_values), dtype=np.bool_)
    if len(qualifying):
        rejected[order[: qualifying[-1] + 1]] = True
    return rejected


def cohens_d(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    if len(first) < 2 or len(second) < 2:
        raise ValueError("each group requires at least two samples")
    pooled = np.sqrt(
        ((len(first) - 1) * first.var(ddof=1) + (len(second) - 1) * second.var(ddof=1))
        / (len(first) + len(second) - 2)
    )
    return float((first.mean() - second.mean()) / pooled)
