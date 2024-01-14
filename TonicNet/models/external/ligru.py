import torch
import torch.nn as nn
import torch.nn.functional as F


class LiGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiGRUCell, self).__init__()

        self.hidden_size = hidden_size

        # Initialize weights
        self.wz = nn.Parameter(torch.Tensor(input_size + hidden_size, 4 * hidden_size))
        nn.init.xavier_uniform_(self.wz)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x: torch.Tensor, h=None):
        batch_size = x.size(0)

        # Initialize hidden state if not provided
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Reshape x to 2D if it's not already
        x = x.view(batch_size, -1)

        # Stack and apply weights
        stack = torch.cat((x, h), dim=1)
        out_gates = stack @ self.wz

        # Calculate gates and hidden states
        z = torch.sigmoid(
            self.bn1(out_gates[:, : self.hidden_size])
            + out_gates[:, self.hidden_size : 2 * self.hidden_size]
        )
        h_tilde = torch.relu(
            self.bn2(out_gates[:, 2 * self.hidden_size : 3 * self.hidden_size])
            + out_gates[:, 3 * self.hidden_size :]
        )

        h_next = z * h + (1 - z) * h_tilde

        return h_next


class LiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.5):
        super(LiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList(
            [
                LiGRUCell(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        # x is now expected to have shape [batch_size, sequence_length, input_size]
        batch_size, sequence_length, _ = x.size()

        # Initialize the hidden state for each layer
        hidden_states = [
            torch.zeros(batch_size, self.hidden_size, device=x.device)
            for _ in range(self.num_layers)
        ]

        # Process the input sequence one time step at a time
        for t in range(sequence_length):
            input_t = x[:, t, :]
            for i, layer in enumerate(self.layers):
                hidden_states[i] = layer(input_t, hidden_states[i])
                input_t = hidden_states[i]

                # Apply dropout except for the last layer and when the model is in evaluation mode
                if i < self.num_layers - 1 and self.training:
                    input_t = F.dropout(input_t, p=self.dropout)

        return hidden_states, hidden_states[-1]


if __name__ == "__main__":
    model = LiGRU(100, 50, num_layers=5, dropout=0.3)
    samples = torch.zeros(32, 10, 100)  # batch_size, seq_len, input_size

    print(model)
    print(model(samples))
