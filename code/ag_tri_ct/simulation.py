from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SimulatedCohort:
    skills: NDArray[np.int64]
    responses: NDArray[np.int64]
    disability: NDArray[np.bool_]


def simulate_cohort(
    trainees: int,
    events: int,
    concepts: int = 800,
    seed: int = 2026,
    disability_fraction: float = 0.1,
) -> SimulatedCohort:
    if trainees not in {200, 500, 1000}:
        raise ValueError("cohort size must match the evaluation protocol")
    generator = np.random.default_rng(seed)
    difficulty = generator.normal(0.0, 0.8, concepts)
    ability = generator.normal(0.0, 1.0, trainees)
    disability = generator.random(trainees) < disability_fraction
    skills = generator.integers(0, concepts, size=(trainees, events), dtype=np.int64)
    responses = np.zeros((trainees, events), dtype=np.int64)
    for event in range(events):
        learning = 0.04 * event
        logits = ability[:, None] + learning - difficulty[skills[:, event]][:, None]
        probabilities = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        responses[:, event] = generator.binomial(1, probabilities)
        ability += 0.02 * responses[:, event]
    return SimulatedCohort(skills, responses, disability)
