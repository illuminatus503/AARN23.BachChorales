import torch
import torch.nn as nn

import tensorly as tl
from tensorly.decomposition import tensor_train_matrix

tl.backend("torch")

from typing import *


class TensorizedGRUCell(nn.RNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_size, hidden_size, bias, num_chunks=3, **factory_kwargs)

    def factorize(self, rank="same", factorization="cp"):
        # ! Factorize the GRU weight
        # Get the tensorized shape for weight_ih (input + hidden)
        tensor_ih_shape = tltorch.utils.get_tensorized_shape(
            *self.weight_ih.shape, min_dim=2
        )
        weight_ih = tltorch.TensorizedTensor.new(
            tensor_ih_shape,
            rank=rank,
            factorization=factorization,
        )
        weight_ih.init_from_matrix(self.weight_ih)
        self.weight_ih = nn.Parameter(weight_ih)

        # ! Factorize the GRU weight_hh
        # Get the tensorized shape for weight_hh (hidden + hidden)
        tensor_hh_shape = tltorch.utils.get_tensorized_shape(
            *self.weight_hh.shape, min_dim=2
        )

        weight_hh = tltorch.TensorizedTensor.new(
            tensor_hh_shape, rank=rank, factorization=factorization
        )
        weight_hh.init_from_matrix(self.weight_hh)
        self.weight_hh = nn.Parameter(weight_hh)

    def forward(
        self, input: torch.Tensor, hx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if input.dim() not in (1, 2):
            raise ValueError(
                f"TensorizedGRUCell: Expected input to be 1D or 2D, got {input.dim()}D instead"
            )
        if hx is not None and hx.dim() not in (1, 2):
            raise ValueError(
                f"TensorizedGRUCell: Expected hidden to be 1D or 2D, got {hx.dim()}D instead"
            )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        ret = self.gru_cell(input, hx)

        if not is_batched:
            ret = ret.squeeze(0)

        return ret

    def gru_cell(self, x, hx):
        x = x.view(-1, x.size(1))

        gate_x = F.linear(x, self.weight_ih, self.bias_ih)
        gate_h = F.linear(hx, self.weight_hh, self.bias_hh)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hx - newgate)

        return hy
