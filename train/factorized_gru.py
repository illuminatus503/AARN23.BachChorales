import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch

from typing import *


class LinearlyFactorizedGRUCell(nn.RNNCellBase):
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

        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.factorized = False

    def factorize(
        self,
        rank="same",
        factorization="cp",
        implementation="factorized",
        checkpointing=True,
    ):
        # Generate inner linear layers
        _linear_ih = nn.Linear(
            in_features=self.input_size,
            out_features=3 * self.hidden_size,
            bias=self.bias,
            device=self.device,
            dtype=self.dtype,
        )
        _linear_ih.weight = self.weight_ih
        _linear_ih.bias = self.bias_ih

        _linear_hh = nn.Linear(
            in_features=self.hidden_size,
            out_features=3 * self.hidden_size,
            bias=self.bias,
            device=self.device,
            dtype=self.dtype,
        )
        _linear_hh.weight = self.weight_hh
        _linear_hh.bias = self.bias_hh

        # Init the IH weights
        self.factor_weight_ih = tltorch.FactorizedLinear.from_linear(
            _linear_ih,
            rank=rank,
            auto_tensorize=True,
            bias=self.bias,
            factorization=factorization,
            implementation=implementation,
            checkpointing=checkpointing,
        )

        # Init the HH weights
        self.factor_weight_hh = tltorch.FactorizedLinear.from_linear(
            _linear_hh,
            rank=rank,
            auto_tensorize=True,
            bias=self.bias,
            factorization=factorization,
            implementation=implementation,
            checkpointing=checkpointing,
        )

        # Set the layer as factorized
        self.factorized = True

    def __gru_cell(self, x, hx):
        x = x.view(-1, x.size(1))

        if self.factorized:
            gate_x = self.factor_weight_ih(x)
            gate_h = self.factor_weight_hh(hx)
        else:
            gate_x = F.linear(x, self.weight_ih, self.bias_ih)
            gate_h = F.linear(hx, self.weight_hh, self.bias_hh)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hx - newgate)

        return hy

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

        # Run the GRU cell
        ret = self.__gru_cell(input, hx)

        if not is_batched:
            ret = ret.squeeze(0)

        return ret


class GRUBase(nn.RNNBase):
    """A Base module for GRU. Inheriting from GRUBase enables compatibility with torch.compile."""

    def __init__(self, *args, **kwargs):
        super().__init__("GRU", *args, **kwargs)


for attr in nn.GRU.__dict__:
    if attr != "__init__":
        setattr(GRUBase, attr, getattr(nn.GRU, attr))


class LinearlyFactorizedGRU(GRUBase):
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

        self.device = device
        self.dtype = dtype
        self.factorized = False

    def __factorize_single_layer(
        self,
        weight_ih: torch.Tensor,
        bias_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_hh: torch.Tensor,
        rank: Union[int, float],
        factorization: str,
        checkpointing: bool,
        implementation: str = "factorized",
    ):
        # Generate inner linear layers
        _linear_ih = nn.Linear(
            in_features=self.input_size,
            out_features=3 * self.hidden_size,
            bias=self.bias,
            device=self.device,
            dtype=self.dtype,
        )
        _linear_ih.weight = weight_ih
        _linear_ih.bias = bias_ih

        _linear_hh = nn.Linear(
            in_features=self.hidden_size,
            out_features=3 * self.hidden_size,
            bias=self.bias,
            device=self.device,
            dtype=self.dtype,
        )
        _linear_hh.weight = weight_hh
        _linear_hh.bias = bias_hh

        # Init the IH weights
        factor_weight_ih = tltorch.FactorizedLinear.from_linear(
            _linear_ih,
            rank=rank,
            auto_tensorize=True,
            bias=self.bias,
            factorization=factorization,
            implementation=implementation,
            checkpointing=checkpointing,
        )

        # Init the HH weights
        factor_weight_hh = tltorch.FactorizedLinear.from_linear(
            _linear_hh,
            rank=rank,
            auto_tensorize=True,
            bias=self.bias,
            factorization=factorization,
            implementation=implementation,
            checkpointing=checkpointing,
        )

        return factor_weight_ih, factor_weight_hh

    def factorize(
        self,
        rank="same",
        factorization="cp",
        implementation="factorized",
        checkpointing=False,
    ):
        self.rank = rank
        self.factorization = factorization
        self.implementation = implementation
        self.checkpointing = checkpointing

        self.factor_linear_ih = []
        self.factor_linear_hh = []

        for layer in range(self.num_layers):
            weights = self._all_weights[layer]
            _factor_linear_ih, _factor_linear_hh = self.__factorize_single_layer(
                weight_ih=getattr(self, weights[0]),
                bias_ih=getattr(self, weights[2]),
                weight_hh=getattr(self, weights[1]),
                bias_hh=getattr(self, weights[3]),
                rank=rank,
                factorization=factorization,
                checkpointing=checkpointing,
                implementation=implementation,
            )

            # Guardamos lineales que representen las puertas
            self.factor_linear_ih.append(_factor_linear_ih)
            self.factor_linear_hh.append(_factor_linear_hh)

        # Se ha factorizado la capa GRU
        self.factorized = True

    @staticmethod
    def _gru_cell(
        x,
        hx,
        weight_ih,
        bias_ih,
        weight_hh,
        bias_hh,
    ):
        x = x.view(-1, x.size(1))

        gate_x = F.linear(x, weight_ih, bias_ih)
        gate_h = F.linear(hx, weight_hh, bias_hh)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hx - newgate)

        return hy

    @staticmethod
    def _factorized_gru_cell(
        x,
        hx,
        factor_linear_ih: tltorch.FactorizedLinear,
        factor_linear_hh: tltorch.FactorizedLinear,
    ):
        x = x.view(-1, x.size(1))

        gate_x = factor_linear_ih(x)
        gate_h = factor_linear_hh(hx)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

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

    def _factorized_gru(self, x, hx):
        if not self.batch_first:
            x = x.permute(
                1, 0, 2
            )  # Change (seq_len, batch, features) to (batch, seq_len, features)

        h_t = list(hx.unbind(0))

        outputs = []
        for x_t in x.unbind(1):
            for layer in range(self.num_layers):
                h_t[layer] = self._factorized_gru_cell(
                    x_t,
                    h_t[layer],
                    self.factor_linear_ih[layer],
                    self.factor_linear_hh[layer],
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

        # Run GRU layer
        if self.factorized:
            result = self._factorized_gru(input, hx)
        else:
            result = self._gru(input, hx)

        output = result[0]
        hidden = result[1]

        return output, hidden


class FactorizedGRU(GRUBase):
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
        rank: Union[int, float],
        order: int,
        factorization: str,
    ):
        # Factorizar el peso GRU (weight_ih)
        tensor_ih_shape = tltorch.utils.get_tensorized_shape(
            in_features=weight_ih.shape[0],
            out_features=weight_ih.shape[1],
            min_dim=2,
            order=order,
        )
        _weight_ih = tltorch.TensorizedTensor.from_matrix(
            weight_ih, *tensor_ih_shape, rank, factorization
        )
        _weight_ih = nn.Parameter(weight_ih)

        # Factorizar el peso GRU (weight_hh)
        tensor_hh_shape = tltorch.utils.get_tensorized_shape(
            in_features=weight_hh.shape[0],
            out_features=weight_hh.shape[1],
            min_dim=2,
            order=order,
        )
        _weight_hh = tltorch.TensorizedTensor.from_matrix(
            weight_hh, *tensor_hh_shape, rank, factorization
        )
        _weight_hh = nn.Parameter(weight_hh)

        return _weight_ih, _weight_hh

    def factorize(
        self,
        rank="same",
        factorization="blocktt",
        order=3,
        implementation="factorized",
        checkpointing=False,
    ):
        self.implementation = implementation
        self.checkpointing = checkpointing
        self.tensor_dropout = tltorch.tensor_hooks.TensorDropout(
            self.dropout, drop_test=False
        )

        for layer in range(self.num_layers):
            # Retrieve weights' names
            weights = self._all_weights[layer]

            _weight_ih, _weight_hh = self.__factorize_single_layer(
                getattr(self, weights[0]),
                getattr(self, weights[1]),
                rank,
                order,
                factorization,
            )

            setattr(self, weights[0], _weight_ih)
            setattr(self, weights[1], _weight_hh)

    @staticmethod
    def _gru_cell(
        x,
        hx,
        weight_ih,
        bias_ih,
        weight_hh,
        bias_hh,
    ):
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
