"""Flux-style flow transformer for audio latents: N1 MMDiT blocks then N2 DiT blocks.

Shapes:
  noisy         [B, C, T]     interpolated latent
  timestep      [B]           continuous in [0, 1]
  context       [B, Tc, Dctx] sequence conditioning, may be None when mmdit_blocks == 0
  aligned_cond  [B, Da, T]    frame-aligned control, one frame per latent frame
  global_cond   [B, Dg]       vector conditioning, may be None
  returns       [B, C, T]     predicted velocity
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_embedding(values: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Embed a continuous [B] timestep in [0, 1]. Scaled by 1000 for resolution at small t."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=values.device, dtype=torch.float32) / max(half, 1)
    )
    args = values.float().unsqueeze(1) * 1000.0 * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; cos/sin: [T, D]
    return x * cos.unsqueeze(0).unsqueeze(0) + _rotate_half(x) * sin.unsqueeze(0).unsqueeze(0)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return apply_rope_single(q, cos, sin), apply_rope_single(k, cos, sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq: int = 2048, base: float = 10000.0):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE head dim must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq = max_seq
        self._build_cache(max_seq)

    def _build_cache(self, length: int) -> None:
        t = torch.arange(length, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq = length

    def forward(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if length > self.max_seq:
            self._build_cache(length)
        return self.cos_cached[:length], self.sin_cached[:length]


def rope_segment(
    q: torch.Tensor, k: torch.Tensor, rope: RotaryEmbedding, start: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to a stream occupying positions [start, start + T) of the joint sequence."""
    length = q.size(2)
    cos, sin = rope(start + length)
    return apply_rope(q, k, cos[start:], sin[start:])


def rope_segment_single(x: torch.Tensor, rope: RotaryEmbedding, start: int) -> torch.Tensor:
    """Same, for a stream that contributes keys but no queries."""
    length = x.size(2)
    cos, sin = rope(start + length)
    return apply_rope_single(x, cos[start:], sin[start:])


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class ConvMLP(nn.Module):
    """SwiGLU MLP with 1-D convs for local temporal structure (audio stream)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Conv1d(dim, hidden, 3, padding=1)
        self.w2 = nn.Conv1d(dim, hidden, 3, padding=1)
        self.out = nn.Conv1d(hidden, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        return self.out(F.silu(self.w1(h)) * self.w2(h)).transpose(1, 2)


class LinearMLP(nn.Module):
    """SwiGLU MLP with linear layers (context stream)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.out = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.w1(x)) * self.w2(x))


class AttentionProjections(nn.Module):
    """Per-stream QKV and output projections with RMSNorm on q/k.

    A stream that is read by the other stream but never written back (the final MMDiT context
    stream) needs neither queries nor an output projection: build it with queries=False, which
    drops q, q_norm, and out. Allocating them would leave parameters with no gradient.
    """

    def __init__(self, dim: int, heads: int, queries: bool = True):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim)
        self.q = nn.Linear(dim, dim, bias=False) if queries else None
        self.q_norm = RMSNorm(self.head_dim) if queries else None
        self.out = nn.Linear(dim, dim, bias=False) if queries else None

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def project_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_norm(self._heads(self.k(x))), self._heads(self.v(x))

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.q is None or self.q_norm is None:
            raise RuntimeError("This stream was built with queries=False")
        k, v = self.project_kv(x)
        return self.q_norm(self._heads(self.q(x))), k, v

    def project_out(self, x: torch.Tensor, length: int) -> torch.Tensor:
        if self.out is None:
            raise RuntimeError("This stream was built with queries=False")
        return self.out(x.transpose(1, 2).reshape(x.size(0), length, -1))


class AdaLNModulation(nn.Module):
    """Shift/scale/gate for attention and MLP (6 * dim). Zero-init so blocks start as identity.

    chunks=2 yields shift/scale only, for a stream that takes no gated residual.
    """

    def __init__(self, dim: int, cond_dim: int, chunks: int = 6):
        super().__init__()
        self.chunks = chunks
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, chunks * dim))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.proj(cond).chunk(self.chunks, dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FlowBlock(nn.Module):
    """MMDiT / DiT block in three configurations.

    multimodal=True                 Flux-style joint attention; both streams update.
    cross_attend=True               single-stream with cross-attention to context, as in
                                    midisynth's LatentDenoiser. For stacks with no MMDiT.
                                    Queries and context keys share one RoPE phase origin.
    neither                         pure self-attention; context is ignored.

    update_context=False reduces the context stream to keys and values only, so the block is a
    joint self/cross attention over audio queries. Use it for the last MMDiT block, whose context
    output nothing reads; otherwise its queries, output projection, MLP, and gates get no gradient.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float,
        multimodal: bool,
        update_context: bool = True,
        cross_attend: bool = False,
    ):
        super().__init__()
        if multimodal and cross_attend:
            raise ValueError("A block is either multimodal (joint attention) or cross-attending")
        self.multimodal = multimodal
        self.cross_attend = cross_attend
        self.update_context = update_context if multimodal else False
        hidden = int(dim * mlp_ratio)
        self.audio_norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.audio_norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.audio_attn = AttentionProjections(dim, heads)
        self.audio_mlp = ConvMLP(dim, hidden)
        # 9 chunks when cross-attending: self-attn, cross-attn, and MLP each need shift/scale/gate.
        self.audio_mod = AdaLNModulation(dim, dim, chunks=9 if cross_attend else 6)
        if multimodal:
            self.ctx_norm1 = nn.LayerNorm(dim, elementwise_affine=False)
            self.ctx_attn = AttentionProjections(dim, heads, queries=self.update_context)
            self.ctx_mod = AdaLNModulation(dim, dim, chunks=6 if self.update_context else 2)
            if self.update_context:
                self.ctx_norm2 = nn.LayerNorm(dim, elementwise_affine=False)
                self.ctx_mlp = LinearMLP(dim, hidden)
        if cross_attend:
            self.cross_norm = nn.LayerNorm(dim, elementwise_affine=False)
            self.cross_q = nn.Linear(dim, dim, bias=False)
            self.cross_kv = nn.Linear(dim, 2 * dim, bias=False)
            self.cross_out = nn.Linear(dim, dim, bias=False)
            self.cross_heads = heads
            self.cross_head_dim = dim // heads
            self.cross_q_norm = RMSNorm(self.cross_head_dim)
            self.cross_k_norm = RMSNorm(self.cross_head_dim)

    def _cross_attention(
        self, audio: torch.Tensor, context: torch.Tensor, rope: RotaryEmbedding,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attend audio queries to context keys, both phased over [0, T).

        Sharing one phase origin is what makes this comparable to the aligned-sum path: p2p's
        context is frame-aligned with the latent, so query i and key i are the same moment in
        time, and position-blind cross-attention would have to rediscover that from content.
        Nothing here assumes equal lengths — a shorter context simply occupies [0, Tc). In p2p,
        context is always built at the same T as audio, so `key_mask` (the audio stream's own
        padding mask) doubles as the context's padding mask too.
        """
        batch, length, _ = audio.shape
        heads, head_dim = self.cross_heads, self.cross_head_dim
        q = self.cross_q(audio).view(batch, length, heads, head_dim).transpose(1, 2)
        k, v = self.cross_kv(context).chunk(2, dim=-1)
        k = k.view(batch, context.size(1), heads, head_dim).transpose(1, 2)
        v = v.view(batch, context.size(1), heads, head_dim).transpose(1, 2)
        q = rope_segment_single(self.cross_q_norm(q), rope, 0)
        k = rope_segment_single(self.cross_k_norm(k), rope, 0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        return self.cross_out(out.transpose(1, 2).reshape(batch, length, -1))

    def forward(
        self,
        audio: torch.Tensor,
        context: torch.Tensor | None,
        cond: torch.Tensor,
        rope: RotaryEmbedding,
        audio_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``audio_mask``, when given, is ``[B, T]`` boolean (True = real frame, False = padding).

        Keys only — a padded position's own query output is simply never read downstream (masked
        out of the loss, never decoded past the item's true length), so masking it as a query too
        would be extra work for no behavioural difference. Reshaped once per block to SDPA's
        ``[B, 1, 1, T]`` broadcast shape.
        """
        key_mask = audio_mask[:, None, None, :] if audio_mask is not None else None
        audio_mod = self.audio_mod(cond)
        a_shift1, a_scale1, a_gate1, a_shift2, a_scale2, a_gate2 = audio_mod[:6]
        t_audio = audio.size(1)

        if self.multimodal and context is not None:
            # Dead code here: validate_config rejects scratch.mmdit_blocks > 0, so this branch
            # never runs and does not need mask support.
            ctx_mod = self.ctx_mod(cond)
            audio_h = modulate(self.audio_norm1(audio), a_shift1, a_scale1)
            ctx_h = modulate(self.ctx_norm1(context), ctx_mod[0], ctx_mod[1])
            qa, ka, va = self.audio_attn.project(audio_h)
            # Distinct positional phases: audio at [0, T), context at [T, T + Tc).
            qa, ka = rope_segment(qa, ka, rope, 0)
            if self.update_context:
                qc, kc, vc = self.ctx_attn.project(ctx_h)
                qc, kc = rope_segment(qc, kc, rope, t_audio)
                query = torch.cat([qa, qc], dim=2)
            else:
                kc, vc = self.ctx_attn.project_kv(ctx_h)
                kc = rope_segment_single(kc, rope, t_audio)
                query = qa
            out = F.scaled_dot_product_attention(
                query, torch.cat([ka, kc], dim=2), torch.cat([va, vc], dim=2)
            )
            audio_out, ctx_out = out[:, :, :t_audio], out[:, :, t_audio:]
            audio = audio + a_gate1.unsqueeze(1) * self.audio_attn.project_out(audio_out, t_audio)
            audio = audio + a_gate2.unsqueeze(1) * self.audio_mlp(
                modulate(self.audio_norm2(audio), a_shift2, a_scale2)
            )
            if self.update_context:
                _, _, c_gate1, c_shift2, c_scale2, c_gate2 = ctx_mod
                context = context + c_gate1.unsqueeze(1) * self.ctx_attn.project_out(ctx_out, context.size(1))
                context = context + c_gate2.unsqueeze(1) * self.ctx_mlp(
                    modulate(self.ctx_norm2(context), c_shift2, c_scale2)
                )
            return audio, context

        audio_h = modulate(self.audio_norm1(audio), a_shift1, a_scale1)
        q, k, v = self.audio_attn.project(audio_h)
        q, k = rope_segment(q, k, rope, 0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        audio = audio + a_gate1.unsqueeze(1) * self.audio_attn.project_out(out, t_audio)

        if self.cross_attend:
            if context is None:
                raise ValueError("A cross-attending block requires context")
            c_shift, c_scale, c_gate = audio_mod[6:9]
            audio = audio + c_gate.unsqueeze(1) * self._cross_attention(
                modulate(self.cross_norm(audio), c_shift, c_scale), context, rope, key_mask
            )

        audio = audio + a_gate2.unsqueeze(1) * self.audio_mlp(
            modulate(self.audio_norm2(audio), a_shift2, a_scale2)
        )
        return audio, context


class FlowTransformer(nn.Module):
    """Predicts rectified-flow velocity for audio latents.

    How a context sequence reaches the audio depends on mmdit_blocks:
      mmdit_blocks > 0   joint attention in the MMDiT stage, then pure self-attention DiT.
      mmdit_blocks == 0  cross-attention to context in every DiT block (midisynth style).
    Pass context_dim=None for a stack conditioned only on global_cond.

    aligned_cond_dim adds a third, separate path for conditioning that is already frame-aligned
    with the latent: it is projected and summed into the audio stream before the blocks, so every
    layer reads it through the residual stream and no attention has to learn the alignment. In p2p
    this is the output of a ``cond.py`` encoder, which is also what feeds context when
    ``model.cond_injection`` selects cross-attention instead.
    """

    def __init__(
        self,
        latent_channels: int,
        hidden_dim: int,
        num_heads: int,
        mmdit_blocks: int,
        dit_blocks: int,
        context_dim: int | None = None,
        global_cond_dim: int | None = None,
        aligned_cond_dim: int | None = None,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if mmdit_blocks > 0 and context_dim is None:
            raise ValueError("context_dim is required when mmdit_blocks > 0")
        if dit_blocks < 1 and mmdit_blocks < 1:
            raise ValueError("Need at least one block")
        # Without MMDiT, context can only reach the audio by cross-attention. Failing to wire
        # it would silently drop the conditioning, so it is enabled rather than ignored.
        cross_attend = mmdit_blocks == 0 and context_dim is not None
        if cross_attend and dit_blocks < 1:
            raise ValueError("context_dim with mmdit_blocks=0 requires dit_blocks >= 1 to cross-attend")
        self.cross_attend = cross_attend
        self.hidden_dim = hidden_dim
        self.audio_proj = nn.Conv1d(latent_channels, hidden_dim, 1)
        self.context_proj = nn.Linear(context_dim, hidden_dim) if context_dim else None
        # Summing a separate projection is identical to concatenating the channels and projecting
        # once, and keeps the unconditional path (aligned_cond zeroed for CFG) a plain no-op.
        self.aligned_proj = nn.Conv1d(aligned_cond_dim, hidden_dim, 1) if aligned_cond_dim else None
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.global_mlp = (
            nn.Sequential(nn.Linear(global_cond_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            if global_cond_dim
            else None
        )
        self.rope = RotaryEmbedding(hidden_dim // num_heads)
        # The last MMDiT block does not update context: nothing downstream reads it.
        self.blocks = nn.ModuleList(
            [
                FlowBlock(
                    hidden_dim, num_heads, mlp_ratio, multimodal=True,
                    update_context=index < mmdit_blocks - 1,
                )
                for index in range(mmdit_blocks)
            ]
            + [
                FlowBlock(
                    hidden_dim, num_heads, mlp_ratio, multimodal=False, cross_attend=cross_attend
                )
                for _ in range(dit_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        nn.init.zeros_(self.final_mod[-1].weight)
        nn.init.zeros_(self.final_mod[-1].bias)
        self.final_proj = nn.Linear(hidden_dim, latent_channels)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        aligned_cond: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``key_padding_mask``, when given, is ``[B, T]`` boolean (True = real frame) matching
        ``noisy``'s own T — a curriculum batch padded with silence past each item's own length.
        Unused (None) for the fixed-clip path, where every item in a batch is already the same
        real length."""
        if context is None and self.context_proj is not None:
            raise ValueError("This model was built with context_dim; pass context or rebuild without it")
        if aligned_cond is None and self.aligned_proj is not None:
            raise ValueError("This model was built with aligned_cond_dim; pass aligned_cond")
        audio = self.audio_proj(noisy).transpose(1, 2)
        if aligned_cond is not None and self.aligned_proj is not None:
            if aligned_cond.size(-1) != noisy.size(-1):
                raise ValueError(
                    f"aligned_cond has {aligned_cond.size(-1)} frames but the latent has {noisy.size(-1)}"
                )
            audio = audio + self.aligned_proj(aligned_cond.to(dtype=audio.dtype)).transpose(1, 2)
        ctx = self.context_proj(context) if (context is not None and self.context_proj) else None

        cond = self.time_mlp(sinusoidal_embedding(timestep, self.hidden_dim).to(dtype=audio.dtype))
        if global_cond is not None and self.global_mlp is not None:
            cond = cond + self.global_mlp(global_cond.to(dtype=audio.dtype))

        for block in self.blocks:
            wants_context = block.multimodal or block.cross_attend
            audio, ctx = block(
                audio, ctx if wants_context else None, cond, self.rope, key_padding_mask
            )

        shift, scale = self.final_mod(cond).chunk(2, dim=-1)
        audio = modulate(self.final_norm(audio), shift, scale)
        out = self.final_proj(audio).transpose(1, 2)
        assert out.shape == noisy.shape, f"{out.shape} != {noisy.shape}"
        return out
