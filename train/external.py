import itertools as it

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.nn.utils.rnn import PackedSequence

import tltorch

from typing import *


class GRUBase(nn.RNNBase):
    """A Base module for GRU. Inheriting from GRUBase enables compatibility with torch.compile."""

    def __init__(self, *args, **kwargs):
        return super().__init__("GRU", *args, **kwargs)


for attr in nn.GRU.__dict__:
    if attr != "__init__":
        setattr(GRUBase, attr, getattr(nn.GRU, attr))


class TensorizedGRU(GRUBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        if bidirectional:
            raise NotImplementedError(
                "Bidirectional LSTMs are not supported yet in this implementation."
            )

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=False,
            device=device,
            dtype=dtype,
        )

    def __factorize_single_layer(
        self,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        rank: int | float,
        order: int,
        factorization: str,
    ):
        # ! Factorize the GRU weight
        # Get the tensorized shape for weight_ih (input + hidden)
        tensor_ih_shape = tltorch.utils.get_tensorized_shape(
            *weight_ih.shape, min_dim=2, order=order
        )
        _weight_ih = tltorch.FactorizedTensor.new(
            tensor_ih_shape, rank=rank, factorization=factorization
        )
        # _weight_ih.init_from_matrix(weight_ih)
        _weight_ih = nn.Parameter(weight_ih)

        # ! Factorize the GRU weight_hh
        # Get the tensorized shape for weight_hh (hidden + hidden)
        tensor_hh_shape = tltorch.utils.get_tensorized_shape(
            *weight_hh.shape, min_dim=2
        )

        _weight_hh = tltorch.FactorizedTensor.new(
            tensor_hh_shape, rank=rank, factorization=factorization
        )
        # _weight_hh.init_from_matrix(weight_hh)
        _weight_hh = nn.Parameter(weight_hh)

        return [_weight_ih, _weight_hh]

    def factorize(
        self,
        rank="same",
        n_tensorized_modes=3,
        factorization="cp",
        implementation="reconstructed",
        checkpointing=False,
    ):
        self.implementation = implementation
        self.checkpointing = checkpointing

        for layer in range(self.num_layers):
            # Retrieve weights' names
            weights = self._all_weights[layer]

            # Factorize those weights
            weight_ih = getattr(self, weights[0])
            weight_hh = getattr(self, weights[1])
            _weights = self.__factorize_single_layer(
                weight_ih,
                weight_hh,
                rank=rank,
                factorization=factorization,
                order=n_tensorized_modes,
            )

            # Set factorized weights
            setattr(self, weights[0], _weights[0])
            setattr(self, weights[1], _weights[1])

    @staticmethod
    def _gru_cell(
        x,
        hx,
        weight_ih,
        bias_ih,
        weight_hh,
        bias_hh,
        in_features,
        hidden_features,
        implementation,
    ):
        x = x.view(-1, x.size(1))

        gate_x = F.linear(x, weight_ih, bias_ih)
        gate_h = F.linear(hx, weight_hh, bias_hh)
        # gate_x = tltorch.functional.factorized_linear(
        #     x, weight_ih, bias_ih, in_features, implementation
        # )
        # gate_h = tltorch.functional.factorized_linear(
        #     hx, weight_hh, bias_hh, hidden_features, implementation
        # )

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = (i_r + h_r).sigmoid()
        inputgate = (i_i + h_i).sigmoid()
        newgate = (i_n + (resetgate * h_n)).tanh()

        hy = newgate + inputgate * (hx - newgate)

        return hy

    def _gru(self, x, hx):
        if not self.batch_first:
            x = x.permute(
                1, 0, 2
            )  # Change (seq_len, batch, features) to (batch, seq_len, features)

        bs, seq_len, input_size = x.size()
        h_t = list(hx.unbind(0))

        weight_ih = []
        weight_hh = []
        bias_ih = []
        bias_hh = []
        for layer in range(self.num_layers):
            # Retrieve weights
            weights = self._all_weights[layer]
            weight_ih.append(getattr(self, weights[0]))
            weight_hh.append(getattr(self, weights[1]))
            if self.bias:
                bias_ih.append(getattr(self, weights[2]))
                bias_hh.append(getattr(self, weights[3]))
            else:
                bias_ih.append(None)
                bias_hh.append(None)

        outputs = []

        for x_t in x.unbind(1):
            for layer in range(self.num_layers):
                h_t[layer] = self._gru_cell(
                    x_t,
                    h_t[layer],
                    weight_ih[layer],
                    bias_ih[layer],
                    weight_hh[layer],
                    bias_hh[layer],
                    self.input_size,
                    self.hidden_size,
                    self.implementation,
                )

                # TODO: checkpointing

                # Apply dropout if in training mode and not the last layer
                if layer < self.num_layers - 1 and self.dropout:
                    x_t = F.dropout(h_t[layer], p=self.dropout, training=self.training)
                else:
                    x_t = h_t[layer]

            outputs.append(x_t)

        outputs = torch.stack(outputs, dim=1)
        if not self.batch_first:
            outputs = outputs.permute(
                1, 0, 2
            )  # Change back (batch, seq_len, features) to (seq_len, batch, features)

        return outputs, torch.stack(h_t, 0)

    def forward(self, input, hx=None):  # noqa: F811
        if input.dim() != 3:
            raise ValueError(
                f"GRU: Expected input to be 3D, got {input.dim()}D instead"
            )
        if hx is not None and hx.dim() != 3:
            raise RuntimeError(
                f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
            )
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        if hx is None:
            hx = torch.zeros(
                self.num_layers,
                max_batch_size,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )

        self.check_forward_args(input, hx, batch_sizes=None)
        result = self._gru(input, hx)

        output = result[0]
        hidden = result[1]

        return output, hidden


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
