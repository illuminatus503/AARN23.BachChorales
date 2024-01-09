import torch
import torch.nn as nn
import torch.nn.functional as F
import tltorch


class TensorizedGRU(nn.Module):
    @classmethod
    def from_gru(cls, gru: nn.GRU, rank="same", factorization="tt"):
        instance = cls(
            gru.input_size,
            gru.hidden_size,
            gru.num_layers,
            gru.bias,
            gru.batch_first,
            gru.dropout,
            gru.bidirectional,
        )

        # Reset parameters
        instance.weight_ih.clear()
        instance.weight_hh.clear()
        instance.bias_ih.clear()
        instance.bias_hh.clear()

        # Add GRU layers
        for layer in range(gru.num_layers):
            weights = gru._all_weights[layer]

            instance.weight_ih.append(
                tltorch.FactorizedTensor.from_tensor(
                    getattr(gru, weights[0]), rank, factorization
                )
            )
            instance.weight_hh.append(
                tltorch.FactorizedTensor.from_tensor(
                    getattr(gru, weights[1]), rank, factorization
                )
            )

            if gru.bias:
                instance.bias_ih.append(getattr(gru, weights[2]))
                instance.bias_hh.append(getattr(gru, weights[3]))
            else:
                instance.bias_ih.append(None)
                instance.bias_hh.append(None)

        return instance

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        if bidirectional:
            raise NotImplementedError(
                "Bidirectional LSTMs are not supported yet in this implementation."
            )

        super(TensorizedGRU, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout

        self.weight_ih, self.bias_ih = [], []
        self.weight_hh, self.bias_hh = [], []

    def _gru_cell(self, x, hx, layer):
        x = x.view(-1, x.size(1))

        # gate_x = F.linear(x, weight_ih, bias_ih)
        # gate_h = F.linear(hx, weight_hh, bias_hh)

        print(self.weight_ih[layer])

        gate_x = tltorch.functional.factorized_linear(
            x,
            self.weight_ih[layer],
            self.bias_ih[layer],
        )

        gate_h = tltorch.functional.factorized_linear(
            hx,
            self.weight_ih[layer],
            self.bias_ih[layer],
        )

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

        h_t = list(hx.unbind(0))

        outputs = []
        for x_t in x.unbind(1):
            for layer in range(self.num_layers):
                h_t[layer] = self._gru_cell(x_t, h_t[layer], layer)

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
                self.hidden_features,
                dtype=input.dtype,
                device=input.device,
            )

        result = self._gru(input, hx)

        output = result[0]
        hidden = result[1]

        return output, hidden


def main():
    gru = nn.GRU(
        input_size=10,
        hidden_size=2,
        num_layers=2,
        batch_first=True,
        dropout=0.1,
    )

    x = torch.randn(1, 1, 10)
    tensorgru = TensorizedGRU.from_gru(gru, rank=0.1)

    x, h = tensorgru(x)

    print(tensorgru)


if __name__ == "__main__":
    main()
