"""Loading ACE-Step v1.5, and hanging the piano condition off it.

ACE-Step's DiT already carries the mechanism this project needs. Its `proj_in` is a strided Conv1d
over `in_channels = 192`, and the forward pass builds that width itself:

    hidden_states = cat([context_latents, x_t], -1)          # 128 + 64
    context_latents = cat([src_latents, chunk_masks], -1)    #  64 + 64

`src_latents` is a **frame-aligned 25 Hz, 64-channel conditioning sequence**. Upstream it holds the
VAE latents of a reference recording (cover, repaint, vocal-to-BGM) and, when there is no
reference, a tile of the checkpoint's own `silence_latent`. Here it holds the output of the
conditioning encoder. Declaring nothing and modifying nothing is the whole of the architectural
change: the condition arrives through the pathway the backbone was pretrained on.

Three details of the upstream code shape everything below.

* `AceStepConditionGenerationModel.prepare_condition` is decorated `@torch.no_grad()`. Routing the
  trainable encoder's output through it would silently sever its gradient, so `model.py` calls
  `.encoder(...)` for the text/lyric/timbre pack and assembles `context_latents` itself. `split_
  condition` here is the seam that makes that possible without forking upstream code.
* `is_covers` selects an LM-hint branch that runs the audio tokenizer and detokenizer over the
  input. It is 0 for every item here, so those two submodules never execute — and, since they are
  never trained either, `trainable_backbone_parameters` leaves them out.
* `chunk_masks` is `[B, T]` upstream and repeated to the latent width before use: 1 = generate,
  0 = preserve the source. Every frame of every training window is generated, so it is all ones.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

# The instruction and prompt template the checkpoint was trained under, copied from ACE-Step-1.5's
# `acestep/constants.py`. They are part of the text distribution, not a stylistic choice: a prompt
# in a different shape is off-distribution for a frozen text path.
DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""


@contextmanager
def _residual_fsq_on_cpu():
    """Keep ACE-Step's quantizer constants off ``meta`` during Transformers model loading.

    Transformers 5.16 always constructs pretrained models on the meta device; its legacy
    ``low_cpu_mem_usage`` argument is ignored.  ``vector_quantize_pytorch.ResidualFSQ`` is not
    meta-safe: its constructor calls ``.item()`` for codebook sizes and registers several
    non-persistent buffers which therefore cannot be supplied later by the checkpoint.

    Only the audio tokenizer owns a ``ResidualFSQ``.  Nothing in this project ever runs it —
    ``is_covers`` is 0 for every item — but the failure is at *construction*, not at call time:
    ``ResidualFSQ.__init__`` calls ``.item()`` on a meta tensor and dies before any forward pass
    exists to avoid.  Constructing this small submodule in an inner CPU device context leaves the 2.39B
    backbone on the memory-efficient meta path while ensuring every quantizer buffer is real and
    can follow the model to its eventual device.  The temporary patch is restored even if loading
    fails.
    """
    from vector_quantize_pytorch import ResidualFSQ

    original_init = ResidualFSQ.__init__

    def cpu_init(self, *args, **kwargs):
        with torch.device("cpu"):
            original_init(self, *args, **kwargs)

    ResidualFSQ.__init__ = cpu_init
    try:
        yield
    finally:
        ResidualFSQ.__init__ = original_init


def load_dit(cfg: DictConfig, device: str | torch.device = "cpu", dtype: torch.dtype | None = None):
    """`AceStepConditionGenerationModel`, with its own modelling code, checked against the config."""
    from transformers import AutoModel

    with _residual_fsq_on_cpu():
        model = AutoModel.from_pretrained(
            str(cfg.ace.model_name),
            trust_remote_code=True,
            dtype=dtype if dtype is not None else torch.float32,
            device_map=None,
        )
    check_backbone_matches(cfg, model.config)
    model.to(device)
    return model


def check_backbone_matches(cfg: DictConfig, model_config: Any) -> None:
    """The config declares the backbone's numbers; the checkpoint is the authority on them.

    Every one of these is load-bearing. `in_channels` decides whether `src_latents` is a slot that
    exists at all; `audio_acoustic_hidden_dim` is the width the conditioning encoder must emit;
    `patch_size` is what a window length has to be a multiple of.
    """
    actual = {
        "latent_channels": int(model_config.audio_acoustic_hidden_dim),
        "in_channels": int(model_config.in_channels),
        "patch_size": int(model_config.patch_size),
        "prompt_dim": int(model_config.text_hidden_dim),
    }
    declared = {key: int(getattr(cfg.ace, key)) for key in actual}
    if actual != declared:
        raise ValueError(
            f"ace.* disagrees with {cfg.ace.model_name}'s own config: declared {declared}, "
            f"checkpoint says {actual}"
        )
    expected_in = 3 * actual["latent_channels"]
    if actual["in_channels"] != expected_in:
        raise ValueError(
            f"{cfg.ace.model_name} takes in_channels={actual['in_channels']}, but this project "
            f"assumes cat([src_latents, chunk_masks, x_t]) = 3 x {actual['latent_channels']} = "
            f"{expected_in}. The conditioning slot is not where it is expected to be."
        )


def load_silence_latent(cfg: DictConfig) -> torch.Tensor:
    """The checkpoint's own `[1, T, 64]` "no reference audio" latent.

    Upstream tiles it into `src_latents` for text2music and feeds a 750-frame slice of it as the
    null reference-audio timbre. Both uses are kept here, and it is also the bias the conditioning
    encoder is initialised at, so an untrained model reproduces stock ACE-Step exactly.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(str(cfg.ace.model_name), "silence_latent.pt")
    latent = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(latent, dict):
        latent = next(iter(latent.values()))
    latent = latent.float()
    if latent.dim() == 2:
        latent = latent.unsqueeze(0)
    channels = int(cfg.ace.latent_channels)
    # The file ships channels-first, `[1, 64, 15000]` — 600 s of it. Everything downstream of
    # `prepare_condition` is time-major, so it is transposed once here rather than at each use.
    if latent.dim() == 3 and latent.shape[1] == channels and latent.shape[-1] != channels:
        latent = latent.transpose(1, 2)
    if latent.dim() != 3 or latent.shape[-1] != channels:
        raise ValueError(
            f"silence_latent.pt has shape {tuple(latent.shape)}, expected [1, T, {channels}]"
        )
    return latent.contiguous()


def tile_silence(silence: torch.Tensor, frames: int) -> torch.Tensor:
    """`[1, T0, 64]` -> `[1, frames, 64]`, repeating rather than padding with zeros.

    Zeros are not the "no audio" point of this latent space — the checkpoint ships a specific
    vector for that, and a window longer than the shipped one has to keep speaking it.
    """
    available = silence.shape[1]
    if available >= frames:
        return silence[:, :frames]
    repeats = -(-frames // available)
    return silence.repeat(1, repeats, 1)[:, :frames]


def load_vae(cfg: DictConfig, device: str | torch.device = "cpu"):
    """The Oobleck VAE, on its own. Held outside any module tree so it never enters a checkpoint."""
    from diffusers import AutoencoderOobleck

    vae = AutoencoderOobleck.from_pretrained(
        str(cfg.ace.assets_repo), subfolder=str(cfg.ace.vae_subfolder)
    )
    rate = int(getattr(vae.config, "sampling_rate", 0))
    if rate != int(cfg.ace.sample_rate):
        raise ValueError(f"ace.sample_rate={cfg.ace.sample_rate} but the VAE runs at {rate}")
    hop = int(getattr(vae, "hop_length", 0))
    if hop != int(cfg.ace.hop_length):
        raise ValueError(f"ace.hop_length={cfg.ace.hop_length} but the VAE hops {hop}")
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    vae.eval().to(device)
    return vae


def load_text_encoder(cfg: DictConfig, device: str | torch.device = "cpu"):
    """Qwen3-Embedding-0.6B and its tokenizer, frozen. Prep only — training reads the cache.

    ACE-Step's `CondEncoder.text_projector` is a `Linear(1024 -> 2048)` baked into the checkpoint,
    so the text encoder is architecturally locked to this one; it is not a swappable choice.
    """
    from transformers import AutoModel, AutoTokenizer

    name, subfolder = str(cfg.ace.assets_repo), str(cfg.ace.text_encoder_subfolder)
    tokenizer = AutoTokenizer.from_pretrained(name, subfolder=subfolder)
    encoder = AutoModel.from_pretrained(name, subfolder=subfolder, dtype=torch.float32)
    width = int(encoder.config.hidden_size)
    if width != int(cfg.ace.prompt_dim):
        raise ValueError(f"ace.prompt_dim={cfg.ace.prompt_dim} but the text encoder is {width}-wide")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval().to(device)
    return tokenizer, encoder


# ACE-Step's `TASK_INSTRUCTIONS`. The `# Instruction` block is not decoration — it is how the
# checkpoint is told which *task* it is doing, and each string is one the model was trained under.
# Using text2music's instruction while handing the model source audio states one task and supplies
# another, which is what produced noise from the first covers-mode renders.
TASK_INSTRUCTIONS = {
    "text2music": "Fill the audio semantic mask based on the given conditions:",
    "cover": "Generate audio semantic tokens based on the given conditions:",
    "repaint": "Repaint the mask area based on the given conditions:",
    # `{}` takes the track classes to add, e.g. "drums, bass, guitar, strings".
    "complete": "Complete the input track with {}:",
}


def dit_prompt(caption: str, metas: str = "", instruction: str | None = None) -> str:
    """A caption in the shape the checkpoint's text path was trained on.

    `instruction` defaults to text2music's. Any task that supplies source audio needs its own —
    see `TASK_INSTRUCTIONS`.
    """
    return SFT_GEN_PROMPT.format(instruction or DEFAULT_DIT_INSTRUCTION, caption, metas)


def backbone_module_names(model) -> dict[str, str]:
    """A one-line map of the submodules this project does and does not touch, for the banner."""
    return {
        "decoder": "the 24-layer DiT — adapted by LoRA, or trained whole",
        "encoder": "text / lyric / timbre condition encoder — trained only in phase=full",
        "tokenizer": "audio tokenizer — never runs, is_covers is always 0",
        "detokenizer": "audio detokenizer — never runs, is_covers is always 0",
    }


def unused_parameter_names(model) -> list[str]:
    """Parameters no forward pass in this project can reach.

    `tokenize`/`detokenize` sit behind the `is_covers` branch, which is 0 for every item here.
    Leaving them in an optimizer would have AdamW carry moments for weights that never receive a
    gradient — 0 harm to the result, but real memory, and a misleading trainable-parameter count.
    """
    names = []
    for prefix in ("tokenizer.", "detokenizer."):
        names.extend(name for name, _ in model.named_parameters() if name.startswith(prefix))
    return names


def read_model_config(cfg: DictConfig) -> dict[str, Any]:
    """The checkpoint's `config.json`, from the Hub cache, without building the model."""
    from huggingface_hub import hf_hub_download

    with open(Path(hf_hub_download(str(cfg.ace.model_name), "config.json"))) as handle:
        return json.load(handle)
