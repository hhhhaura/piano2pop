"""Precompute the one fixed text condition every item is trained and sampled under.

The model is conditioned on the piano roll alone; ACE-Step's cross-attention always sees the same
prompt, the checkpoint's `SFT_GEN_PROMPT` template around "instrumental, no vocals", together with
the empty lyric block. Both are encoded once by Qwen3-Embedding-0.6B and cached, so training never
loads a 0.6B encoder — not in the main process, and not in any dataloader worker.

    .cache/prompt/L<prompt_max_length>/<corpus>-null.npz
        lyric      [L, 1024] float16   the empty-lyric block
        lyric_mask [L]        bool
        text       [prompt_max_length, 1024] float16   the fixed prompt
        text_mask  [prompt_max_length]       bool

The prompt is padded to a fixed length, because `AceStepDiTModel.forward` discards the attention
mask it is given. The lyric block is still fed rather than skipped: the condition encoder packs
lyric, timbre and text into one cross-attention sequence, and an absent branch changes that
packing. Upstream takes lyric embeddings from the encoder's **embedding table** rather than running
the encoder over them (`conditioning_embed.py::infer_lyric_embeddings`), which is reproduced
exactly.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from .ace import dit_prompt, load_text_encoder
from .config import TrainConfig, config_dir, validate_config
from .paths import null_prompt_path
from .stages import atomic_output

INSTRUMENTAL_SUFFIX = ", instrumental, no vocals"
# `# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>`, from ACE-Step's `_format_lyrics`.
LYRIC_TEMPLATE = "# Languages\n{}\n\n# Lyric\n{}<|endoftext|>"
LYRIC_LANGUAGE = "en"
# The `# Metas` block. Empty: the corpus carries no per-track metadata of the kind ACE-Step's
# planner emits, and inventing one would put the frozen text path off-distribution.
METAS = ""


def decorate(caption: str) -> str:
    text = " ".join(str(caption).split()).rstrip(" .,;")
    return f"{text}{INSTRUMENTAL_SUFFIX}" if text else INSTRUMENTAL_SUFFIX.lstrip(", ")


@torch.no_grad()
def embed(tokenizer, encoder, texts: list[str], length: int, device: str):
    """Tokenize to a fixed length and take `last_hidden_state`, as ACE-Step's pipeline does."""
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=int(length),
        return_tensors="pt",
    )
    ids = inputs.input_ids.to(device)
    mask = inputs.attention_mask.to(device)
    hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    return (
        hidden.to(torch.float16).cpu().numpy(),
        mask.to(torch.bool).cpu().numpy(),
        [int(row.sum()) for row in inputs.attention_mask],
    )


@torch.no_grad()
def embed_lyric(tokenizer, encoder, text: str, device: str):
    """The lyric branch: the raw embedding table, not the encoder. Matches upstream exactly."""
    inputs = tokenizer(text, padding="longest", truncation=True, max_length=2048, return_tensors="pt")
    ids = inputs.input_ids.to(device)
    hidden = encoder.embed_tokens(ids)
    return hidden.to(torch.float16).cpu().numpy(), inputs.attention_mask.to(torch.bool).numpy()


def build(cfg: DictConfig, *, device: str) -> dict[str, str]:
    tokenizer, encoder = load_text_encoder(cfg, device)
    length = int(cfg.ace.prompt_max_length)
    lyric, lyric_mask = embed_lyric(
        tokenizer, encoder, LYRIC_TEMPLATE.format(LYRIC_LANGUAGE, ""), device
    )
    text, text_mask, _ = embed(tokenizer, encoder, [dit_prompt(decorate(""), METAS)], length, device)
    path = null_prompt_path(cfg)
    with atomic_output(path) as temporary:
        # np.savez appends `.npz` to a path that does not end in it, which would defeat the atomic
        # rename; writing through the handle keeps the temporary name intact.
        with open(temporary, "wb") as handle:
            np.savez(
                handle, lyric=lyric[0], lyric_mask=lyric_mask[0], text=text[0], text_mask=text_mask[0]
            )
    return {"written": str(path)}


def load_null_prompt(cfg: DictConfig) -> dict[str, torch.Tensor]:
    """The cached lyric and text blocks, each with a leading batch dimension of 1."""
    path = null_prompt_path(cfg)
    if not path.is_file():
        raise RuntimeError(f"No null prompt at {path}; run `p2pa-prep-text` first")
    with np.load(path) as stored:
        return {
            "lyric": torch.from_numpy(np.asarray(stored["lyric"])).float()[None],
            "lyric_mask": torch.from_numpy(np.asarray(stored["lyric_mask"])).bool()[None],
            "text": torch.from_numpy(np.asarray(stored["text"])).float()[None],
            "text_mask": torch.from_numpy(np.asarray(stored["text_mask"])).bool()[None],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("overrides", nargs="*", help="Hydra-style config overrides")
    args = parser.parse_args()

    ConfigStore.instance().store(name="p2pa_schema", node=TrainConfig)
    with initialize_config_dir(version_base=None, config_dir=str(config_dir())):
        cfg = compose(config_name="config", overrides=list(args.overrides))
    validate_config(cfg)

    print(json.dumps(build(cfg, device=args.device), sort_keys=True))


if __name__ == "__main__":
    main()
