import torch
import numpy as np
from typing import Tuple, Union
import pytest
import torch.nn.functional as F
import math

from zhilight.internals_ import functions


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    if cos.dim() == 2:
        cos = cos[: x.size(-2), :]
        sin = sin[: x.size(-2), :]
    elif cos.dim() == 3:
        cos = cos[:, : x.size(-2), :]
        sin = sin[:, : x.size(-2), :]
    elif cos.dim() == 4:
        cos = cos[:, :, : x.size(-2), :]
        sin = sin[:, :, : x.size(-2), :]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbeddingESM(torch.nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(
        self,
        dim: int,
        base: Union[int, float] = 10000,
        distance_scale: Union[int, float] = 1,
        dtype=torch.half,
    ):
        super().__init__()
        self.base = base
        self.distance_scale = distance_scale
        self.dtype = dtype

        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq.to(dtype))

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

        self.apply_rotary_pos_emb = apply_rotary_pos_emb

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.size(seq_dimension)
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            if x.dim() == 2:
                self._cos_cached = emb.cos()
                self._sin_cached = emb.sin()
            elif x.dim() == 3:
                self._cos_cached = emb.cos()[None, :, :]
                self._sin_cached = emb.sin()[None, :, :]
            elif x.dim() == 4:
                self._cos_cached = emb.cos()[None, None, :, :]
                self._sin_cached = emb.sin()[None, None, :, :]
        return self._cos_cached, self._sin_cached

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(
            k, seq_dimension=-2
        )
        return (
            self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


@pytest.mark.parametrize("SIZE", [(2, 4, 2)])  # seqlen, num_head, dim_head
@pytest.mark.parametrize("BATCH", [1, 2, 4])
def test_rotary_embedding(SIZE, BATCH):
    rtol, atol = (1e-3, 3e-4)

    if BATCH == 1:
        pos = torch.arange(0, SIZE[0], dtype=torch.int32, device="cuda")
        h_q = torch.randn((SIZE[0], SIZE[1], SIZE[2]), dtype=torch.half, device="cuda")
        h_k = torch.randn((SIZE[0], SIZE[1], SIZE[2]), dtype=torch.half, device="cuda")
    else:
        pos = torch.arange(0, SIZE[0], dtype=torch.int32, device="cuda").repeat((BATCH, 1))
        h_q = torch.randn(
            (BATCH, SIZE[0], SIZE[1], SIZE[2]), dtype=torch.half, device="cuda"
        )
        h_k = torch.randn(
            (BATCH, SIZE[0], SIZE[1], SIZE[2]), dtype=torch.half, device="cuda"
        )

    ff_pt = RotaryEmbeddingESM(SIZE[2], dtype=torch.half).cuda()

    if BATCH == 1:
        out_q, out_k = functions.rotary_embedding_2(
            SIZE[2],
            pos.cpu().numpy(),
            h_q.view(SIZE[0], SIZE[1] * SIZE[2]).cpu().numpy(),
            h_k.view(SIZE[0], SIZE[1] * SIZE[2]).cpu().numpy(),
        )
    else:
        out_q, out_k = functions.rotary_embedding_2(
            SIZE[2],
            pos.cpu().numpy(),
            h_q.view(-1, SIZE[0], SIZE[1] * SIZE[2]).cpu().numpy(),
            h_k.view(-1, SIZE[0], SIZE[1] * SIZE[2]).cpu().numpy(),
        )

    if BATCH == 1:
        out_q_pt, out_k_pt = ff_pt.forward(h_q.permute(1, 0, 2), h_k.permute(1, 0, 2))
        out_q_pt = out_q_pt.permute(1, 0, 2).view(SIZE[0], SIZE[1] * SIZE[2])
        out_k_pt = out_k_pt.permute(1, 0, 2).view(SIZE[0], SIZE[1] * SIZE[2])

    else:
        out_q_pt, out_k_pt = ff_pt.forward(
            h_q.permute(0, 2, 1, 3), h_k.permute(0, 2, 1, 3)
        )
        out_q_pt = out_q_pt.permute(0, 2, 1, 3).view(-1, SIZE[0], SIZE[1] * SIZE[2])
        out_k_pt = out_k_pt.permute(0, 2, 1, 3).view(-1, SIZE[0], SIZE[1] * SIZE[2])
    print(out_q)
    print(out_q_pt)
    assert torch.allclose(
        torch.from_numpy(out_q).to(torch.half), out_q_pt.cpu(), rtol=rtol, atol=atol
    )


if __name__ == "__main__":
    test_rotary_embedding((5, 4, 4), 2)
