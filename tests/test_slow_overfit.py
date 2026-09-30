from __future__ import annotations

import pytest
import torch

from p2pa.sample import sample_latent
from tests.conftest import make_batch


@pytest.mark.slow
def test_one_batch_overfit_proves_training_and_sampling_time_agree(tiny_module, tiny_cfg):
    """The cheap end-to-end guard against integrating a correctly trained flow backwards."""
    torch.manual_seed(42)
    tiny_module.cfg_ratio = 0.0
    tiny_module.backbone_frozen = False
    for layer in tiny_module.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0.0
    for name, parameter in tiny_module.named_parameters():
        parameter.requires_grad_(not name.startswith(("backbone.tokenizer.", "backbone.detokenizer.")))

    batch = make_batch(tiny_cfg, frames=8, batch=1)
    fixed_noise = torch.randn_like(batch["latent"])
    fixed_timestep = torch.full((1,), 0.5)
    tiny_module.eval()
    with torch.no_grad():
        initial_loss = tiny_module.flow_loss(
            batch, timestep=fixed_timestep, noise=fixed_noise
        ).item()

    optimizer = torch.optim.Adam(
        [parameter for parameter in tiny_module.parameters() if parameter.requires_grad], lr=2e-3
    )
    tiny_module.train()
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = tiny_module.flow_loss(batch)
        loss.backward()
        optimizer.step()

    tiny_module.eval()
    with torch.no_grad():
        final_loss = tiny_module.flow_loss(
            batch, timestep=fixed_timestep, noise=fixed_noise
        ).item()
        generated = sample_latent(
            tiny_module, batch, steps=50, guidance=1.0, seed=123
        )

    generator = torch.Generator().manual_seed(123)
    initial_noise = torch.randn(batch["latent"].shape, generator=generator)
    target_mse = torch.mean((generated - batch["latent"]) ** 2).item()
    noise_mse = torch.mean((initial_noise - batch["latent"]) ** 2).item()
    print(
        f"overfit fixed_loss={initial_loss:.6f}->{final_loss:.6f} "
        f"sample/noise_mse={target_mse / noise_mse:.6f}"
    )

    assert final_loss < initial_loss / 10
    assert target_mse < noise_mse * 0.1
