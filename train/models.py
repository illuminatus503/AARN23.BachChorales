import torch
import torch.nn as nn

from train.external import VariationalDropout
from train.factorized_gru import FactorizedGRU

import tltorch

"""
File containing classes representing various Neural architectures
"""


class TonicNet(nn.Module):
    def __init__(
        self,
        nb_tags,
        nb_layers=1,
        z_dim=0,
        nb_rnn_units=100,
        batch_size=1,
        seq_len=1,
        dropout=0.0,
    ):
        super(TonicNet, self).__init__()

        self.nb_layers = nb_layers
        self.nb_rnn_units = nb_rnn_units
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dropout = dropout
        self.z_dim = z_dim
        self.z_emb_size = 32

        self.nb_tags = nb_tags

        # build actual NN
        self.__build_model()

    def __build_model(self):
        self.embedding = nn.Embedding(self.nb_tags, self.nb_rnn_units)
        self.z_embedding = nn.Embedding(80, self.z_emb_size)
        # *Unused* but key exists in state_dict
        self.pos_emb = nn.Embedding(64, 0)

        self.dropout_i = VariationalDropout(
            max(0.0, self.dropout - 0.2), batch_first=True
        )

        # design RNN (save params. for later)
        self.input_size = self.nb_rnn_units
        if self.z_dim > 0:
            self.input_size += self.z_emb_size

        # OLD GRU network (encoder)
        # self.rnn = nn.GRU(
        #     input_size=self.input_size,
        #     hidden_size=self.nb_rnn_units,
        #     num_layers=self.nb_layers,
        #     batch_first=True,
        #     dropout=self.dropout,
        # )

        self.rnn = FactorizedGRU(
            input_size=self.input_size,
            hidden_size=self.nb_rnn_units,
            num_layers=self.nb_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        self.dropout_o = VariationalDropout(self.dropout, batch_first=True)

        # output layer which projects back to tag space
        self.hidden_to_tag = nn.Linear(self.input_size, self.nb_tags, bias=False)

    def __init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_rnn_units)
        hidden_a = torch.randn(self.nb_layers, self.batch_size, self.nb_rnn_units)

        if torch.cuda.is_available():
            hidden_a = hidden_a.cuda()

        return hidden_a

    def factorize_model(self):
        # ! Tensorization of the embeedding layers
        self.embedding = tltorch.FactorizedEmbedding.from_embedding(
            self.embedding,
            rank=0.1,  # 1/10 of the original params.
        )
        self.z_embedding = tltorch.FactorizedEmbedding.from_embedding(
            self.z_embedding,
            rank=0.1,  # 1/10 of the original params.
        )

        # ! Tensorization of the GRU
        self.rnn.factorize(rank=0.01, factorization="tucker")

        # ! Tensorization of the linear layer
        self.hidden_to_tag = tltorch.FactorizedLinear.from_linear(
            linear=self.hidden_to_tag,
            rank=0.1,  # 1/10 of orig. params.
            # factorization="blocktt",  # default: 'cp'
            # checkpointing=True,  # default: False
        )

    def forward(
        self, X, z=None, train_embedding=True, sampling=False, reset_hidden=True
    ):
        # reset the RNN hidden state.
        if not sampling:
            self.seq_len = X.shape[1]
            if reset_hidden:
                self.hidden = self.__init_hidden()

        self.embedding.weight.requires_grad = train_embedding

        # ---------------------
        # Combine inputs
        X = self.embedding(X)
        X = X.view(self.batch_size, self.seq_len, self.nb_rnn_units)

        # repeating pitch encoding
        if self.z_dim > 0:
            Z = self.z_embedding(z % 80)
            Z = Z.view(self.batch_size, self.seq_len, self.z_emb_size)
            X = torch.cat((Z, X), 2)

        X = self.dropout_i(X)

        # Run through RNN
        X, self.hidden = self.rnn(X, self.hidden)

        if self.z_dim > 0:
            X = torch.cat((Z, X), 2)

        X = self.dropout_o(X)

        # run through linear layer
        X = self.hidden_to_tag(X)

        Y_hat = X
        return Y_hat


# MARK:- Custom s2s Cross Entropy loss
class CrossEntropyTimeDistributedLoss(nn.Module):
    """loss function for multi-timsetep model output"""

    def __init__(self):
        super(CrossEntropyTimeDistributedLoss, self).__init__()

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, y_hat, y):
        _y_hat = y_hat.squeeze(0)
        _y = y.squeeze(0)

        # Loss from one sequence
        loss = self.loss_func(_y_hat, _y)
        loss = torch.sum(loss)
        return loss
