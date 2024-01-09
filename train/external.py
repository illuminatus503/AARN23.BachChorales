import itertools as it

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.nn.utils.rnn import PackedSequence

import tltorch

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

        if torch.cuda.is_available():
            self.weight_ih.to(torch.device("cuda"))
            self.weight_hh.to(torch.device("cuda"))

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


class TensorizedGRU(nn.Module):
    def __init__(self):
        pass


class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension. See
    https://arxiv.org/abs/1512.05287 for more details.

    Note that this is not applied to the recurrent activations in the
    LSTM like the above paper. Instead, it is applied to the inputs
    and outputs of the recurrent layer.
    """

    def __init__(self, dropout: float, batch_first: Optional[bool] = False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.0:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(
                max_batch_size, 1, x.size(2), requires_grad=False
            ).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(
                1, max_batch_size, x.size(2), requires_grad=False
            ).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        self.optimizer = base_optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        for group in self.param_groups:
            group["step_counter"] = 0
        self.slow_weights = [
            [p.clone().detach() for p in group["params"]] for group in self.param_groups
        ]

        for w in it.chain(*self.slow_weights):
            w.requires_grad = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        loss = self.optimizer.step()
        for group, slow_weights in zip(self.param_groups, self.slow_weights):
            group["step_counter"] += 1
            if group["step_counter"] % self.k != 0:
                continue
            for p, q in zip(group["params"], slow_weights):
                if p.grad is None:
                    continue
                q.data.add_(self.alpha, p.data - q.data)
                p.data.copy_(q.data)
        return loss
