import torch
import torch.nn as nn
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor
import numpy as np


class TensorizedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, rank):
        super(TensorizedGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank

        # Inicializar los pesos y factorizarlos
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(3 * hidden_size))
        self.reset_parameters()

        # Descomponer los pesos
        self.factorize_weights()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def factorize_weights(self):
        # Descomposición CP de los pesos
        self.factors_ih = parafac(self.weight_ih.detach().numpy(), rank=self.rank)
        self.factors_hh = parafac(self.weight_hh.detach().numpy(), rank=self.rank)

    def forward(self, input, hx=None):
        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )

        # Reconstruir los pesos desde sus factores
        weight_ih_reconstructed = torch.Tensor(cp_to_tensor(self.factors_ih))
        weight_hh_reconstructed = torch.Tensor(cp_to_tensor(self.factors_hh))

        # GRU forward pass (simplificado)
        gi = torch.mm(input, weight_ih_reconstructed.t()) + self.bias
        gh = torch.mm(hx, weight_hh_reconstructed.t())
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hx - newgate)

        return hy


# Ejemplo de uso
input_size = 10
hidden_size = 5
rank = 3
model = TensorizedGRU(input_size, hidden_size, rank)
input = torch.randn(1, input_size)
output = model(input)

print(output)
