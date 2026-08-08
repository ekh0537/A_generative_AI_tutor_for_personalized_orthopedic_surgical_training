from pathlib import Path

import torch
from ag_tri_ct.training import AtomicCheckpoint, TrainingState, cosine_schedule, set_seed
from torch import nn


def test_training_loss_decreases() -> None:
    set_seed(2026)
    model = nn.Linear(4, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    inputs = torch.randn(32, 4)
    targets = inputs.sum(dim=1, keepdim=True)
    losses = []
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


def test_atomic_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "state.pt"
    checkpoint = AtomicCheckpoint()
    state = TrainingState(3, 40, 2026, 0.86)
    checkpoint.save(path, model, optimizer, state)
    loaded = checkpoint.load(path, model, optimizer)
    assert loaded == state
    assert not path.with_suffix(".pt.tmp").exists()


def test_cosine_schedule_boundaries() -> None:
    assert cosine_schedule(0, 100) == 1.0
    assert cosine_schedule(100, 100) == 0.0
