import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class VolumeRecord:
    identifier: str
    volume_path: Path
    mask_path: Path
    concept_index: int
    fold: int


@dataclass(frozen=True)
class InteractionRecord:
    learner_id: str
    skill: int
    correct: int
    timestamp: int


class VolumeDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, records: list[VolumeRecord], patch_shape: tuple[int, int, int]) -> None:
        self.records = tuple(records)
        self.patch_shape = patch_shape

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        record = self.records[index]
        volume = np.load(record.volume_path).astype(np.float32)
        mask = np.load(record.mask_path).astype(np.int64)
        volume, mask = self._center_crop(volume, mask)
        normalized = (volume - volume.mean()) / max(volume.std(), 1e-6)
        return (
            torch.from_numpy(normalized[None]),
            torch.from_numpy(mask),
            torch.tensor(record.concept_index),
        )

    def _center_crop(
        self, volume: NDArray[np.float32], mask: NDArray[np.int64]
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        slices: list[slice] = []
        for size, target in zip(volume.shape, self.patch_shape, strict=True):
            start = max((size - target) // 2, 0)
            slices.append(slice(start, min(start + target, size)))
        selection = tuple(slices)
        return volume[selection], mask[selection]


def read_volume_manifest(path: str | Path) -> list[VolumeRecord]:
    records: list[VolumeRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            records.append(
                VolumeRecord(
                    identifier=row["identifier"],
                    volume_path=Path(row["volume_path"]),
                    mask_path=Path(row["mask_path"]),
                    concept_index=int(row["concept_index"]),
                    fold=int(row["fold"]),
                )
            )
    return records


def stream_interactions(path: str | Path) -> Iterator[InteractionRecord]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            yield InteractionRecord(
                str(raw["learner_id"]),
                int(raw["skill"]),
                int(raw["correct"]),
                int(raw["timestamp"]),
            )


def cold_start_trajectories(
    records: list[InteractionRecord], maximum_events: int = 10
) -> dict[str, list[InteractionRecord]]:
    learners: dict[str, list[InteractionRecord]] = {}
    for record in sorted(records, key=lambda item: (item.learner_id, item.timestamp)):
        sequence = learners.setdefault(record.learner_id, [])
        if len(sequence) < maximum_events:
            sequence.append(record)
    return learners
