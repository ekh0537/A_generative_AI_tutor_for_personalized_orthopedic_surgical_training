import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TrainingState:
    epoch: int
    step: int
    seed: int
    best_metric: float


class AtomicCheckpoint:
    def save(
        self,
        path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        state: TrainingState,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": state.epoch,
            "step": state.step,
            "seed": state.seed,
            "best_metric": state.best_metric,
        }
        torch.save(payload, temporary)
        os.replace(temporary, destination)

    def load(
        self, path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> TrainingState:
        payload: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        seed = int(payload["seed"])
        set_seed(seed)
        return TrainingState(
            int(payload["epoch"]), int(payload["step"]), seed, float(payload["best_metric"])
        )


def cosine_schedule(step: int, total_steps: int, warmup_steps: int = 0) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))


def clip_gradients(model: nn.Module, maximum_norm: float) -> Tensor:
    return torch.nn.utils.clip_grad_norm_(model.parameters(), maximum_norm)


def prepare_distributed(model: nn.Module, device: torch.device) -> nn.Module:
    model = model.to(device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    return model
