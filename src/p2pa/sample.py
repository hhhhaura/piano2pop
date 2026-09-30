"""Sampling: ACE-Step's own Euler ODE over the rectified flow, and full-song inference.

The solver is a transcription of `AceStepConditionGenerationModel.generate_audio`'s `ode` branch,
not a reimplementation of flow matching in general. `t` runs from 1 (noise) to 0 (data), the DiT
predicts the velocity `v = x1 - x0`, and each step is `x <- x - v * (t_curr - t_prev)`. Guidance is
APG (`apg_forward`), also the checkpoint's own — plain linear CFG at scale 7 over-saturates this
model, which is why upstream ships a momentum-buffered projection instead.

`backbone=scratch` has its own sampler, `_sample_scratch`: the same Euler scheme in that model's
opposite time convention, with plain linear guidance on the roll.

Two inference modes, mirroring `p2p-stable`:

`window`          one window at the length the model was trained on. What the listening exports use.
`glued_full_song` overlapping windows, sampled jointly and blended after every solver step, for a
                  whole song. Blending inside the loop rather than crossfading afterwards is what
                  keeps the seams from being audible: neighbouring windows agree on the trajectory
                  rather than on two finished takes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .audio import encode_mp3, render_midi, require_executable, save_mp3
from .config import latent_fps, latent_frames, migrate_config, resolve_path, validate_config
from .paths import notes_dir
from .pianoroll import cached_notes, crop_notes, piano_roll, write_midi


def _apg(module, cond: torch.Tensor, uncond: torch.Tensor, scale: float, buffer):
    """The checkpoint's own adaptive projected guidance.

    Run inside `torch.cuda.device(...)` because upstream's `project()` ends with

        device_type = v0.device.type          # "cuda", never "cuda:2"
        return v0_parallel.to(dtype).to(device_type), v0_orthogonal.to(dtype).to(device_type)

    and `.to("cuda")` means the *default* device, not the tensor's own. On any GPU but cuda:0 the
    projection therefore comes back on the wrong card and the next line dies with "Expected all
    tensors to be on the same device". Setting the default device for the call is the smallest fix
    that does not fork the checkpoint's guidance; it affects every ACE-Step render with
    `guidance > 1` on a non-default GPU.
    """
    if cond.is_cuda:
        with torch.cuda.device(cond.device):
            return module._remote.apg_forward(
                pred_cond=cond, pred_uncond=uncond, guidance_scale=scale,
                momentum_buffer=buffer, dims=[1],
            )
    return module._remote.apg_forward(
        pred_cond=cond, pred_uncond=uncond, guidance_scale=scale, momentum_buffer=buffer, dims=[1]
    )


@torch.no_grad()
def sample_latent(
    module,
    batch: dict,
    *,
    steps: int = 30,
    guidance: float = 7.0,
    shift: float = 1.0,
    seed: int | None = None,
    blend: "WindowBlend | None" = None,
) -> torch.Tensor:
    """`[B, T, 64]` generated latents for one batch of conditioning."""
    module.eval()
    device = module.device
    # Training gets its autocast from Lightning's `bf16-mixed`; sampling has no Trainer around it,
    # so it has to establish the same context itself. Without this the float32 roll encoder and
    # noise meet a bfloat16 backbone (`phase.backbone_dtype`) and every generation dies with
    # "mat1 and mat2 must have the same dtype" — which is exactly how the first listening panel
    # came back four-for-four empty.
    with _autocast(module, device):
        return _sample_latent(
            module, batch, steps=steps, guidance=guidance, shift=shift, seed=seed, blend=blend,
        )


@contextlib.contextmanager
def _autocast(module, device):
    dtype = next(module.backbone.parameters()).dtype
    if dtype == torch.float32 or str(device).startswith("cpu"):
        yield
        return
    with torch.autocast(device_type="cuda", dtype=dtype):
        yield


@torch.no_grad()
def _sample_latent(
    module,
    batch: dict,
    *,
    steps: int,
    guidance: float,
    shift: float,
    seed: int | None,
    blend: "WindowBlend | None",
) -> torch.Tensor:
    device = module.device
    cond = module.build_conditioning(batch)

    if module.scratch:
        frames = int(batch["mask"].shape[1])
        return _sample_scratch(
            module, cond, frames=frames, steps=steps, guidance=guidance, seed=seed, blend=blend
        )

    context = cond["context_latents"]
    hidden = cond["encoder_hidden_states"]
    mask = cond["encoder_attention_mask"]
    batch_size, frames = context.shape[0], context.shape[1]
    dtype = context.dtype

    do_guidance = guidance > 1.0
    if do_guidance:
        null = module.backbone.null_condition_emb.to(hidden.dtype).expand_as(hidden)
        hidden = torch.cat([hidden, null], dim=0)
        mask = torch.cat([mask, mask], dim=0)
        # Duplicated identically, not nulled: ACE-Step's guidance replaces the *text* pack only,
        # and the roll — riding inside `context_latents` — is never dropped.
        context = torch.cat([context, context], dim=0)

    if seed is None:
        noisy = torch.randn(batch_size, frames, module.latent_channels, device=device, dtype=dtype)
    else:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        noisy = torch.randn(
            batch_size, frames, module.latent_channels,
            generator=generator, device=device, dtype=dtype,
        )

    timeline = torch.linspace(1.0, 0.0, int(steps) + 1, device=device, dtype=dtype)
    if shift != 1.0:
        timeline = shift * timeline / (1 + (shift - 1) * timeline)

    buffer = module._remote.MomentumBuffer()
    for current, following in zip(timeline[:-1], timeline[1:]):
        stacked = torch.cat([noisy, noisy], dim=0) if do_guidance else noisy
        step = current * torch.ones(stacked.shape[0], device=device, dtype=dtype)
        velocity = module.run_decoder(
            stacked,
            step,
            encoder_hidden_states=hidden,
            encoder_attention_mask=mask,
            context_latents=context,
        )
        if do_guidance:
            conditioned, unconditioned = velocity.chunk(2)
            velocity = _apg(module, conditioned, unconditioned, float(guidance), buffer)
        noisy = noisy - velocity * (current - following)
        if blend is not None:
            # Inside the loop, so neighbouring windows agree on a trajectory rather than on two
            # finished takes. Blending only at the end leaves an audible seam at every join.
            noisy = blend(noisy)
    return noisy


def _sample_scratch(module, cond, *, frames, steps, guidance, seed, blend):
    """Euler integration in the scratch backbone's convention, `t: 0 -> 1`, noise to data.

    Guidance is linear: the unconditional branch is the same forward with the roll zeroed, which
    is exactly what `drop_conditioning` substitutes in training. Windows are blended after every
    step through the same `WindowBlend` as the finetune's sampler, so the geometry of a song's
    shifted final window has one description.
    """
    backbone = module.backbone
    reference = cond["aligned_cond"]
    device, dtype = reference.device, reference.dtype
    shape = (reference.shape[0], int(module.cfg.ace.latent_channels), frames)
    if seed is None:
        noisy = torch.randn(shape, device=device, dtype=dtype)
    else:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        noisy = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    null = {key: torch.zeros_like(value) for key, value in cond.items()}
    step_size = 1.0 / max(1, int(steps))
    for index in range(int(steps)):
        t = torch.full((shape[0],), index * step_size, device=device, dtype=dtype)
        velocity = backbone.denoise(noisy, t, cond)
        if guidance > 1.0:
            unconditioned = backbone.denoise(noisy, t, null)
            velocity = unconditioned + guidance * (velocity - unconditioned)
        noisy = noisy + step_size * velocity
        if blend is not None:
            noisy = backbone.to_backbone(blend(backbone.to_project(noisy)))
    return backbone.to_project(noisy)


class WindowBlend:
    """Blend overlapping windows back together after every solver step, with a tapered weight.

    **Tapered, not flat**, and the difference is not cosmetic. The first version of this divided by
    a plain count map, so every frame of an overlap became the unweighted mean of the two windows
    covering it. `p2p`'s `blend_overlaps` instead cross-fades linearly — `w*A + (1-w)*B` with `w`
    ramping 1 to 0 across the overlap — so each window stays authoritative near its own centre and
    only cedes at its edge.

    A flat mean of two partly-independent draws is about 3 dB quieter than either, and with a 10 s
    overlap on 20 s windows *every* frame of a song is in some overlap, so the whole render is
    attenuated and smeared rather than just the seams. Measured on a 316 s song generated from
    `p2p_pop2piano_humanize`'s own 518k-step weights: RMS 0.143 through the flat blend against
    0.254 through p2p's cross-fade, from identical weights. That is the same model sounding thin.

    The taper is derived from the actual start list rather than from a single `overlap` argument,
    because the geometry here is deliberately not uniform: the final window is shifted back to end
    on the song's last frame, so its hop is shorter than the rest. Normalising by the summed
    weights generalises the cross-fade to that layout and to any coverage depth, and reduces to
    exactly p2p's linear cross-fade when the windows are evenly spaced and two deep.
    """

    def __init__(self, starts: list[int], frames: int, total: int, device) -> None:
        self.starts = list(starts)
        self.frames = int(frames)
        self.total = int(total)

        # The regular overlap: `frames` minus the *largest* hop. The shifted final window has a
        # smaller hop and therefore a larger overlap, and taking the max keeps it from setting the
        # taper for the whole song.
        hops = [b - a for a, b in zip(self.starts, self.starts[1:])]
        taper = max(1, frames - max(hops)) if hops else frames

        # A trapezoid: rises across `taper` frames, flat in the middle, falls across `taper`. Never
        # exactly zero, so a frame covered by a single window still normalises to weight 1 instead
        # of 0/0 — which is what makes the song's first and last frames come out at full strength.
        position = torch.arange(frames, device=device, dtype=torch.float32)
        ramp = torch.minimum(position + 1.0, torch.flip(position, [0]) + 1.0)
        self.weight = (ramp / float(taper)).clamp(max=1.0)[:, None]

        norm = torch.zeros(total, device=device)
        for start in self.starts:
            norm[start : start + frames] += self.weight[:, 0]
        self.norm = norm.clamp_min(1e-6)[:, None]

    def join(self, windows: torch.Tensor) -> torch.Tensor:
        """`[W, frames, C]` -> the whole song, `[total, C]`, tapered and normalised."""
        joined = torch.zeros(self.total, windows.shape[-1], device=windows.device, dtype=windows.dtype)
        weight = self.weight.to(joined.dtype)
        for index, start in enumerate(self.starts):
            joined[start : start + self.frames] += weight * windows[index]
        return joined / self.norm.to(joined.dtype)

    def __call__(self, windows: torch.Tensor) -> torch.Tensor:
        """The per-step form: join, then hand each window its slice of the agreed result back."""
        joined = self.join(windows)
        return torch.stack(
            [joined[start : start + self.frames] for start in self.starts], dim=0
        )




@torch.no_grad()
def decode_latent(module, latent: torch.Tensor) -> torch.Tensor:
    """`[B, T, 64]` -> a `[channels, samples]` waveform on the CPU, for the first item.

    Chunked, for the same reason the encode is (`prepare.encode_waveform`): a whole song in one
    pass asks the allocator for 12.5 GB, which on a shared card is an OOM retry at best. The
    overlap is decoded and discarded on both sides so every kept sample comes from a full
    receptive field, and the joins are sample-exact rather than crossfaded.
    """
    vae = module.vae()
    dtype = next(vae.parameters()).dtype
    hop = int(module.cfg.ace.hop_length)
    chunk = int(module.cfg.ace.chunk_frames)
    overlap = int(module.cfg.ace.chunk_overlap_frames)
    # The VAE is channels-first; everything downstream of the DiT is time-major.
    value = latent[:1].transpose(1, 2).to(device=module.device, dtype=dtype)
    frames = value.shape[-1]
    if chunk <= 0 or frames <= chunk:
        return vae.decode(value).sample[0].float().cpu()

    pieces = []
    for start in range(0, frames, chunk):
        stop = min(frames, start + chunk)
        left, right = max(0, start - overlap), min(frames, stop + overlap)
        decoded = vae.decode(value[:, :, left:right]).sample[0]
        head = (start - left) * hop
        pieces.append(decoded[:, head : head + (stop - start) * hop].float().cpu())
    return torch.cat(pieces, dim=-1)


def roll_from_notes(cfg, notes, frames: int, offset: float) -> np.ndarray:
    """The conditioning roll for an arbitrary piano MIDI at inference."""
    from .instrument_groups import normalize_programs

    programs = normalize_programs([int(p) for p in cfg.prep.programs])
    return piano_roll(
        notes,
        frames,
        float(cfg.roll.frames_per_second),
        int(cfg.roll.pitch_low),
        int(cfg.roll.pitch_high),
        velocity_in_onset=bool(cfg.roll.velocity_in_onset),
        offset=offset,
        programs=programs,
    )


def build_batch(cfg, notes, frames: int, offsets: list[float]) -> dict:
    """One conditioning batch: `len(offsets)` windows of `frames` latent frames each."""
    rolls = [
        torch.from_numpy(roll_from_notes(cfg, notes, frames, offset).astype(np.float32))
        for offset in offsets
    ]
    count = len(offsets)
    return {
        "sample_id": [f"window{index}" for index in range(count)],
        "roll": torch.stack(rolls),
        "mask": torch.ones(count, frames, dtype=torch.bool),
        "seconds": torch.full((count,), frames / latent_fps(cfg)),
        "start_seconds": torch.tensor(offsets, dtype=torch.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--midi", required=True, help="the conditioning piano MIDI")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("window", "glued_full_song"), default="window")
    parser.add_argument("--seconds", type=float, default=0.0, help="0 = the trained length")
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from .model import load_checkpoint
    from .train import stored_hyperparameters

    checkpoint = resolve_path(args.checkpoint)
    # The config without the tensors: a full checkpoint is 27 GB, and it is loaded once, below.
    cfg = migrate_config(OmegaConf.create(stored_hyperparameters(checkpoint)))
    validate_config(cfg)

    module = load_checkpoint(cfg, str(checkpoint)).to(args.device).eval()
    steps = int(args.steps) or int(cfg.model.inference_steps)
    guidance = float(args.guidance) or float(cfg.model.guidance)

    notes = cached_notes(args.midi, str(notes_dir(cfg)))
    span = max((note.end for note in notes), default=0.0)

    trained = float(cfg.length.max_seconds)
    fps = latent_fps(cfg)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.mode == "window":
        seconds = float(args.seconds) or trained
        frames = latent_frames(cfg, seconds)
        batch = build_batch(cfg, notes, frames, [0.0])
        latent = sample_latent(
            module, batch, steps=steps, guidance=guidance,
            shift=float(cfg.model.shift), seed=args.seed,
        )
    else:
        frames = latent_frames(cfg, trained)
        hop = max(1, frames - latent_frames(cfg, float(args.overlap_seconds)))
        total = max(frames, int(np.ceil(span * fps)))
        total += (-total) % int(cfg.ace.patch_size)
        starts = list(range(0, max(1, total - frames + 1), hop))
        if starts[-1] + frames < total:
            starts.append(total - frames)
        batch = build_batch(cfg, notes, frames, [s / fps for s in starts])
        blend = WindowBlend(starts, frames, total, module.device)
        windows = sample_latent(
            module, batch, steps=steps, guidance=guidance,
            shift=float(cfg.model.shift), seed=args.seed, blend=blend,
        )
        # The same weighted join the per-step blend uses. Assembling the final latent with a
        # different rule than the one the windows were converged under would undo the agreement
        # they reached — and it is how the flat-average bug survived: two places, one description.
        latent = blend.join(windows)[None]

    rate = int(cfg.ace.sample_rate)
    bitrate = int(cfg.trainer.mp3_bitrate_kbps)
    save_mp3(decode_latent(module, latent), rate, output / "generated.mp3",
             str(cfg.data.ffmpeg), bitrate)

    seconds = float(latent.shape[1]) / fps
    write_midi(crop_notes(notes, 0.0, seconds), output / "piano.mid")
    try:
        require_executable(str(cfg.data.fluidsynth))
        wav = output / "piano.wav"
        render_midi(output / "piano.mid", wav, str(cfg.data.soundfont), rate,
                    fluidsynth=str(cfg.data.fluidsynth))
        encode_mp3(wav, output / "piano.mp3", str(cfg.data.ffmpeg), bitrate)
        wav.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - fluidsynth is optional
        pass

    (output / "index.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "mode": args.mode, "seconds": seconds,
        "steps": steps, "guidance": guidance, "seed": args.seed,
    }, indent=2, sort_keys=True))
    print(f"[p2pa] wrote {output}")


if __name__ == "__main__":
    main()
