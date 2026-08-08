from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LossWeights:
    geometry: float = 0.5
    text: float = 0.3
    knowledge_tracing: float = 0.2
    fragment: float = 0.3

    def validate(self) -> None:
        values = (self.geometry, self.text, self.knowledge_tracing, self.fragment)
        if any(value < 0.0 for value in values):
            raise ValueError("loss weights must be nonnegative")


@dataclass(frozen=True)
class ScoreWeights:
    segmentation: float = 0.3
    text: float = 0.3
    knowledge_tracing: float = 0.3
    hallucination: float = 0.1

    def validate(self) -> None:
        total = self.segmentation + self.text + self.knowledge_tracing + self.hallucination
        if abs(total - 1.0) > 1e-8:
            raise ValueError("score weights must sum to one")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 2026
    device: str = "cuda"
    precision: str = "bf16"
    world_size: int = 8
    epochs: int = 200
    batch_size: int = 16
    gradient_accumulation: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 0
    scheduler: str = "cosine"
    gradient_clip: float = 1.0
    ontology_size: int = 800
    embedding_dim: int = 256
    retrieval_top_k: int = 5
    few_shot_events: int = 10
    bootstrap_samples: int = 10000
    loss: LossWeights = field(default_factory=LossWeights)
    score: ScoreWeights = field(default_factory=ScoreWeights)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation * self.world_size

    def validate(self) -> None:
        if self.ontology_size != 800:
            raise ValueError("ontology size must be 800")
        if self.embedding_dim != 256:
            raise ValueError("embedding dimension must be 256")
        if self.retrieval_top_k != 5:
            raise ValueError("retrieval top-k must be 5")
        if self.few_shot_events > 10:
            raise ValueError("cold-start adaptation cannot exceed 10 events")
        self.loss.validate()
        self.score.validate()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            raw: dict[str, Any] = yaml.safe_load(stream)
        loss = LossWeights(
            geometry=float(raw.pop("loss_geometry")),
            text=float(raw.pop("loss_text")),
            knowledge_tracing=float(raw.pop("loss_knowledge_tracing")),
            fragment=float(raw.pop("loss_fragment")),
        )
        score = ScoreWeights(
            segmentation=float(raw.pop("score_segmentation")),
            text=float(raw.pop("score_text")),
            knowledge_tracing=float(raw.pop("score_knowledge_tracing")),
            hallucination=float(raw.pop("score_hallucination")),
        )
        config = cls(**raw, loss=loss, score=score)
        config.validate()
        return config
