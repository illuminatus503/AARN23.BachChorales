import torch.nn as nn

# Definir una GRU
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2)

# Acceder al state_dict
state_dict = gru.state_dict()

# Imprimir las claves y las dimensiones de los parámetros
for key in state_dict:
    print(f"{key}: {state_dict[key].size()}")
