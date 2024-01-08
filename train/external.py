import itertools as it

import tltorch

import torch
import torch.nn as nn

from torch.optim.optimizer import Optimizer
from torch.nn.utils.rnn import PackedSequence

from typing import *


class TensorizedGRU(nn.GRU):
    @overload
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        ...

    @overload
    def __init__(self, *args, **kwargs):
        ...

    def __init__(self, *args, **kwargs):
        if "proj_size" in kwargs:
            raise ValueError(
                "proj_size argument is only supported for LSTM, not RNN or GRU"
            )
        super().__init__("GRU", *args, **kwargs)

    @classmethod
    def from_gru(cls, gru: nn.GRU, rank=1.0, factorization="cp"):
        layer = cls(
            input_size=gru.input_size,
            hidden_size=gru.hidden_size,
            num_layers=gru.num_layers,
            bias=gru.bias,
            batch_first=gru.batch_first,
            dropout=gru.dropout,
            bidirectional=gru.bidirectional,
        )

        state_dict = gru.state_dict()
        for key in state_dict:
            if "weight" in key:
                with torch.no_grad():
                    state_dict[key] = tltorch.FactorizedTensor.from_tensor(
                        state_dict[key],
                        rank=rank,
                        factorization=factorization,
                    )

        layer.load_state_dict(state_dict)


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
