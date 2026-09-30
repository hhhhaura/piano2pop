from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from p2pa.callbacks import LossSpikeGuard


class LoggingLinear(torch.nn.Linear):
    def __init__(self):
        super().__init__(1, 1)
        self.logged = []

    def log(self, name, value):
        self.logged.append((name, value))


def test_loss_spike_guard_discards_the_optimizer_update():
    guard = LossSpikeGuard(factor=4.0, warmup=2, window=10)
    trainer = SimpleNamespace(global_step=3)
    module = LoggingLinear()
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3, weight_decay=0.1)
    batch = {
        "sample_id": ["bad-track"],
        "seconds": torch.tensor([30.0]),
        "start_seconds": torch.tensor([12.5]),
    }

    for value in (1.0, 1.2):
        guard.on_train_batch_start(trainer, module, batch, 0)
        guard.on_before_backward(trainer, module, torch.tensor(value))
    assert guard.history == pytest.approx([1.0, 1.2])

    for parameter in module.parameters():
        parameter.grad = torch.ones_like(parameter)
    guard.on_train_batch_start(trainer, module, batch, 0)
    guard.on_before_backward(trainer, module, torch.tensor(10.0))
    guard.on_before_optimizer_step(trainer, module, optimizer)

    assert guard.skipped == 1
    assert guard.history == pytest.approx([1.0, 1.2])
    assert all(parameter.grad is None for parameter in module.parameters())
    assert module.logged == [("train/spikes", 1.0)]
