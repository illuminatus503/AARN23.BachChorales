import torch

from train.factorized_gru import LinearlyFactorizedGRUCell, LinearlyFactorizedGRU


def main():
    # gru = LinearlyFactorizedGRUCell(input_size=10, hidden_size=5)
    gru = LinearlyFactorizedGRU(input_size=10, hidden_size=5, num_layers=2, dropout=0.3)
    gru.factorize(rank="same", checkpointing=False)

    x = torch.randn(1, 1, 10)

    print(gru(x))


if __name__ == "__main__":
    main()
