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
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1,
    ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.weight = torch.nn.parameter.Parameter(
            torch.empty((dim_out, dim_in), dtype=dtype)
        )
        torch.nn.init.normal_(self.weight, mean=init_mean, std=init_std)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        x = F.linear(x, self.weight)
        return x


class FeedForward(torch.nn.Module):
    r"""FeedForward module

    Args:
        dim_in (int): input dimension.
        dim_ff (int): middle dimension.
        dim_out (int, optional): output dimension. Defaults to None, which means dim_in = dim_out.
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in feed-forward layer. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in feed-forward layer. Defaults to 0.02.
        bias (bool, optional): whether to use bias term in fully-connected layers used in feed-forward module. Defaults to False.
        activate_fn (str, optional): Defaults to `gated_gelu`.
        dropout_p (int, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        dtype=torch.half,
        dropout_p: Optional[float] = None,
    ):
        super().__init__()

        self.w_in = Linear(
            dim_in=dim_model,
            dim_out=dim_ff,
            dtype=dtype,
        )

        self.w_gated = Linear(
            dim_in=dim_model,
            dim_out=dim_ff,
            dtype=dtype,
        )
        self.act = torch.nn.GELU("tanh")

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.w_out = Linear(
            dim_in=dim_ff,
            dim_out=dim_model,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of feed-forward module.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of feed-forward module.
        """  # noqa: E501
        # gate_score = self.w_in(x)
        print(self.w_in(x))
        gate_score = self.act(self.w_in(x))

        x = self.w_gated(x)

        x = gate_score * x

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.w_out(x)

        return x


@pytest.mark.parametrize("SIZE", [(2, 4)])
@pytest.mark.parametrize("BATCH", [2, 4])
@pytest.mark.parametrize("SEQLEN", [2, 4, 8])
def test_feedforward(SIZE, BATCH, SEQLEN):
    rtol, atol = (1e-3, 1e-2)

    input = torch.randn([BATCH, SEQLEN, SIZE[0]], dtype=torch.half, device="cuda")

    ff = layers.FeedForward(SIZE[0], SIZE[1], "gelu", 0, False, False)
    ff_pt = FeedForward(SIZE[0], SIZE[1], dtype=torch.half).cuda(0)

    state_dict_pt = ff_pt.state_dict(prefix="ff.")
    # transposed tensor must be contiguous before passing to numpy.
    ff.load_state_dict(
        dict(
            [
                (k, v.contiguous().cpu().numpy())
                for k, v in state_dict_pt.items()
            ]
        )
    )

    state_dict = ff.named_parameters()
    for name, param in state_dict_pt.items():
        assert name in state_dict
        assert torch.allclose(
            torch.from_numpy(state_dict[name]).to(torch.half),
            param.cpu(),
            rtol=rtol,
            atol=atol,
        )
    out = ff.forward(input.cpu().numpy())
    # out = ff.forward(np.ndarray(shape=[2,2], dtype=np.half, buffer=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.half)))

    out_pt = ff_pt.forward(input)

    print(out)
    print(out_pt)
    assert torch.allclose(
        torch.from_numpy(out).to(torch.half), out_pt.cpu(), rtol=rtol, atol=atol
    )


if __name__ == "__main__":
    test_feedforward((2, 4), 2, 2)
