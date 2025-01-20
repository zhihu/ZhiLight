import torch
import numpy as np
from typing import Optional
import pytest
import torch.nn.functional as F
import math

from zhilight.internals_ import layers


class Linear(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        act_fn_type: str = "",
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1,
    ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.act_fn_type = act_fn_type
        self.weight = torch.nn.parameter.Parameter(
            torch.empty((dim_out, dim_in), dtype=dtype)
        )
        if act_fn_type.lower() == "gelu":
            self.act = torch.nn.GELU()
        elif act_fn_type.lower() == "silu":
            self.act = torch.nn.SiLU()
        torch.nn.init.normal_(self.weight, mean=init_mean, std=init_std)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        x = F.linear(x, self.weight)

        if self.act_fn_type:
            x = self.act(x)
        return x


@pytest.mark.parametrize("SIZE", [(4, 2)])
@pytest.mark.parametrize("BATCH", [2, 4])
@pytest.mark.parametrize("SEQLEN", [4, 8])
@pytest.mark.parametrize("ACTIVATION", ["", "silu", "gelu"])
@pytest.mark.parametrize("DTYPE", [torch.half, torch.bfloat16])
def test_linear(SIZE, BATCH, SEQLEN, ACTIVATION, DTYPE):
    if DTYPE == torch.bfloat16:
        # TODO should match bfloat16 gemm with pytorch.
        rtol, atol = (1e-2, 3e-3)
    else:
        rtol, atol = (1e-3, 3e-3)

    ff = layers.Linear(
        SIZE[0],
        SIZE[1],
        ACTIVATION,
        False,
        "bfloat" if DTYPE == torch.bfloat16 else "half",
    )
    # ff.init_parameters(1024)
    ff_pt = Linear(SIZE[0], SIZE[1], ACTIVATION, dtype=DTYPE).cuda()

    input = torch.randn([BATCH, SEQLEN, SIZE[0]], dtype=DTYPE, device="cuda")
    input.requires_grad = True

    state_dict = ff_pt.state_dict()
    weight_pt = state_dict["weight"]
    print(state_dict)
    ff.load_state_dict(state_dict)  # shares pytorch's gpu memory.
    weight = ff.named_parameters()["weight"]
    print(torch.amax(weight_pt - weight))

    out = ff.forward(input)
    out_pt = ff_pt.forward(input)
    print(out)
    print(out_pt)
    print(torch.amax(out_pt - out))

    assert torch.allclose(out, out_pt, rtol=rtol, atol=atol)


if __name__ == "__main__":
    test_linear((5, 6), 4, 2, "", torch.bfloat16)
