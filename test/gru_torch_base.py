import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import *


class GRUCell(nn.RNNCellBase):
    r"""A gated recurrent unit (GRU) cell that performs the same operation as nn.LSTMCell but is fully coded in Python.

    .. note::
        This class is implemented without relying on CuDNN, which makes it compatible with :func:`torch.vmap` and :func:`torch.compile`.

    Examples:
        >>> import torch
        >>> from torchrl.modules.tensordict_module.rnn import GRUCell
        >>> device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")
        >>> B = 2
        >>> N_IN = 10
        >>> N_OUT = 20
        >>> V = 4  # vector size
        >>> gru_cell = GRUCell(input_size=N_IN, hidden_size=N_OUT, device=device)

        # single call
        >>> x = torch.randn(B, 10, device=device)
        >>> h0 = torch.zeros(B, 20, device=device)
        >>> with torch.no_grad():
        ...     h1 = gru_cell(x, h0)

        # vectorised call - not possible with nn.GRUCell
        >>> def call_gru(x, h):
        ...     h_out = gru_cell(x, h)
        ...     return h_out
        >>> batched_call = torch.vmap(call_gru)
        >>> x = torch.randn(V, B, 10, device=device)
        >>> h0 = torch.zeros(V, B, 20, device=device)
        >>> with torch.no_grad():
        ...     h1 = batched_call(x, h0)
    """

    __doc__ += nn.GRUCell.__doc__

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

    def forward(
        self, input: torch.Tensor, hx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if input.dim() not in (1, 2):
            raise ValueError(
                f"GRUCell: Expected input to be 1D or 2D, got {input.dim()}D instead"
            )
        if hx is not None and hx.dim() not in (1, 2):
            raise ValueError(
                f"GRUCell: Expected hidden to be 1D or 2D, got {hx.dim()}D instead"
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


# copy GRU
class GRUBase(nn.RNNBase):
    """A Base module for GRU. Inheriting from GRUBase enables compatibility with torch.compile."""

    def __init__(self, *args, **kwargs):
        return super().__init__("GRU", *args, **kwargs)


for attr in nn.GRU.__dict__:
    if attr != "__init__":
        setattr(GRUBase, attr, getattr(nn.GRU, attr))


class GRU(GRUBase):
    """A PyTorch module for executing multiple steps of a multi-layer GRU. The module behaves exactly like :class:`torch.nn.GRU`, but this implementation is exclusively coded in Python.

    .. note::
        This class is implemented without relying on CuDNN, which makes it compatible with :func:`torch.vmap` and :func:`torch.compile`.

    Examples:
        >>> import torch
        >>> from torchrl.modules.tensordict_module.rnn import GRU

        >>> device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")
        >>> B = 2
        >>> T = 4
        >>> N_IN = 10
        >>> N_OUT = 20
        >>> N_LAYERS = 2
        >>> V = 4  # vector size
        >>> gru = GRU(
        ...     input_size=N_IN,
        ...     hidden_size=N_OUT,
        ...     device=device,
        ...     num_layers=N_LAYERS,
        ... )

        # single call
        >>> x = torch.randn(B, T, N_IN, device=device)
        >>> h0 = torch.zeros(N_LAYERS, B, N_OUT, device=device)
        >>> with torch.no_grad():
        ...     h1 = gru(x, h0)

        # vectorised call - not possible with nn.GRU
        >>> def call_gru(x, h):
        ...     h_out = gru(x, h)
        ...     return h_out
        >>> batched_call = torch.vmap(call_gru)
        >>> x = torch.randn(V, B, T, 10, device=device)
        >>> h0 = torch.zeros(V, N_LAYERS, B, N_OUT, device=device)
        >>> with torch.no_grad():
        ...     h1 = batched_call(x, h0)
    """

    __doc__ += nn.GRU.__doc__

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

    @staticmethod
    def _gru_cell(x, hx, weight_ih, bias_ih, weight_hh, bias_hh):
        x = x.view(-1, x.size(1))

        gate_x = F.linear(x, weight_ih, bias_ih)
        gate_h = F.linear(hx, weight_hh, bias_hh)

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
                )

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
