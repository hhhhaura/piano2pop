from __future__ import annotations

import torch
import transformers

from p2pa.ace import _residual_fsq_on_cpu


def test_residual_fsq_is_materialized_inside_transformers_meta_init():
    """Pretrained loading builds the model under ``torch.device('meta')``.

    ResidualFSQ calls ``.item()`` in its constructor and owns non-persistent buffers, so those
    tensors must be real at construction time: checkpoint loading cannot fill them afterwards.
    """
    from vector_quantize_pytorch import ResidualFSQ

    original_init = ResidualFSQ.__init__
    with torch.device("meta"), _residual_fsq_on_cpu():
        quantizer = ResidualFSQ(levels=[8, 5, 5, 5], num_quantizers=2, dim=64)

    assert ResidualFSQ.__init__ is original_init
    assert {tensor.device.type for tensor in quantizer.parameters()} == {"cpu"}
    assert {tensor.device.type for tensor in quantizer.buffers()} == {"cpu"}


def test_transformers_matches_the_checkpoint_attention_implementation():
    """5.x imports the remote code but returns NaNs from its attention internals."""
    assert tuple(int(part) for part in transformers.__version__.split(".")[:2]) == (4, 57)
