from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from p2pa.config import TrainConfig
from p2pa.pianoroll import Note, write_midi


@pytest.fixture
def cfg(tmp_path):
    """Default config, as Hydra would compose it from the dataclass schema."""
    config = OmegaConf.structured(TrainConfig)
    # The project cache may be a read-only external mount in sandboxed test environments.
    config.data.notes_dir = str(tmp_path / "notes")
    config.data.prompt_dir = str(tmp_path / "prompt")
    return config


@pytest.fixture
def tiny_cfg(cfg):
    """Small enough to build and run on CPU inside a test."""
    cfg.model.cond_dim = 32
    cfg.model.cond_layers = 2
    cfg.ace.latent_channels = 8
    cfg.ace.in_channels = 24
    cfg.ace.prompt_dim = 16
    cfg.ace.refer_frames = 10
    cfg.trainer.val_timesteps = 2
    return cfg


def tiny_backbone(cfg):
    """A randomly initialised AceStep model with the real architecture at toy width.

    Built from the checkpoint's own config class and modelling code, so the conditioning
    assembly and the freeze logic are exercised against the real module tree — only the weights
    and the widths are made cheap.
    """
    from transformers import AutoConfig, AutoModel

    # `local_files_only`: the suite must not touch the network. The checkpoint's config and its
    # modelling code are already in the Hub cache after the first real run, and a test that
    # silently depends on a Hub round-trip hangs forever inside a network-restricted sandbox
    # instead of failing — which is exactly how it was found.
    config = AutoConfig.from_pretrained(
        str(cfg.ace.model_name), trust_remote_code=True, local_files_only=True
    )
    config.hidden_size = 128
    config.intermediate_size = 256
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 32
    config.num_lyric_encoder_hidden_layers = 1
    config.num_timbre_encoder_hidden_layers = 1
    config.num_audio_decoder_hidden_layers = 1
    config.num_attention_pooler_hidden_layers = 1
    config.layer_types = ["full_attention", "sliding_attention"]
    config.audio_acoustic_hidden_dim = int(cfg.ace.latent_channels)
    config.in_channels = int(cfg.ace.in_channels)
    config.timbre_hidden_dim = int(cfg.ace.latent_channels)
    config.text_hidden_dim = int(cfg.ace.prompt_dim)
    # `fsq_dim` must equal `hidden_size` — it does in the released config (both 2048), because the
    # tokenizer pools into `hidden_size` and quantises at `fsq_dim`. The toy model had 64 against
    # 128, which no test noticed while `is_covers` was always 0 and the tokenizer therefore never
    # ran. The covers-mode baseline runs it, and it failed as `mat1 and mat2 shapes cannot be
    # multiplied (7x128 and 64x6)` — a broken fixture, not a broken model.
    config.fsq_dim = config.hidden_size
    config.vocab_size = 64
    config.timbre_fix_frame = int(cfg.ace.refer_frames)
    # The checkpoint's config declares bfloat16; a toy model runs on the CPU in float32.
    # `from_config` forwards unknown kwargs to the model's own __init__, so `local_files_only`
    # belongs on the call above and only there. The remote code is already resolved by then.
    model = AutoModel.from_config(config, trust_remote_code=True).float()
    return model


@pytest.fixture
def tiny_module(tiny_cfg, tmp_path, monkeypatch):
    """A real `P2PAModule` around the toy backbone, with a cached null prompt on disk."""
    import p2pa.model as model_module
    from p2pa.paths import null_prompt_path

    monkeypatch.setattr(model_module, "load_dit", lambda cfg, device, dtype=None: tiny_backbone(cfg))
    monkeypatch.setattr(
        model_module,
        "load_silence_latent",
        lambda cfg: torch.full((1, 64, int(cfg.ace.latent_channels)), 0.25),
    )
    target = null_prompt_path(tiny_cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    width = int(tiny_cfg.ace.prompt_dim)
    with open(target, "wb") as handle:
        np.savez(
            handle,
            lyric=np.zeros((5, width), dtype=np.float16),
            lyric_mask=np.ones((5,), dtype=bool),
            text=np.zeros((7, width), dtype=np.float16),
            text_mask=np.ones((7,), dtype=bool),
        )
    return model_module.P2PAModule(tiny_cfg)


def make_batch(cfg, frames: int = 20, batch: int = 2, channels: int | None = None) -> dict:
    from p2pa.pianoroll import total_roll_channels

    roll_channels = channels or total_roll_channels(cfg)
    return {
        "sample_id": [f"track{i}" for i in range(batch)],
        "latent": torch.randn(batch, frames, int(cfg.ace.latent_channels)),
        "roll": torch.rand(batch, roll_channels, frames),
        "mask": torch.ones(batch, frames, dtype=torch.bool),
        "seconds": torch.full((batch,), frames / 25.0),
        "start_seconds": torch.zeros(batch),
    }


def write_notes(path, notes) -> None:
    write_midi(list(notes), path)


def scale_notes(count: int, *, start: float = 0.0, step: float = 0.5) -> list[Note]:
    return [
        Note(pitch=60 + (i % 12), start=start + i * step, end=start + i * step + step * 0.9,
             velocity=80, program=0, is_drum=False)
        for i in range(count)
    ]
