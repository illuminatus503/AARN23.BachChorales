import numpy as np

import torch
import torch.nn as nn

from ..audio import MAX_SEQ


def position_encoding_init(n_position:int, emb_dim:int) -> torch.Tensor:
    """Init the sinusoid position encoding table"""

    # keep dim 0 for padding token position encoding zero vector
    position_enc = torch.tensor(
        [
            [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
            if pos != 0
            else np.zeros(emb_dim)
            for pos in range(n_position)
        ],
        dtype=torch.float32,
    )

    position_enc[1:, 0::2] = np.sin(
        position_enc[1:, 0::2]
    )  # apply sin on 0th,2nd,4th...emb_dim
    
    position_enc[1:, 1::2] = np.cos(
        position_enc[1:, 1::2]
    )  # apply cos on 1st,3rd,5th...emb_dim

    if torch.cuda.is_available():
        position_enc = position_enc.cuda()

    return position_enc


class Transformer_Model(nn.Module):
    def __init__(
        self,
        nb_tags,
        nb_layers=1,
        pe_dim=0,
        emb_dim=100,
        batch_size=1,
        seq_len=MAX_SEQ,
        dropout=0.0,
        encoder_only=True,
    ):
        super(Transformer_Model, self).__init__()

        self.nb_layers = nb_layers
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pe_dim = pe_dim
        self.dropout = dropout

        self.nb_tags = nb_tags

        self.encoder_only = encoder_only

        # build actual NN
        self.__build_model()

    def __build_model(self):
        self.embedding = nn.Embedding(self.nb_tags, self.emb_dim)

        if not self.encoder_only:
            self.embedding2 = nn.Embedding(self.nb_tags, self.emb_dim)

        self.pos_emb = position_encoding_init(MAX_SEQ, self.pe_dim)
        self.pos_emb.requires_grad = False

        self.dropout_i = nn.Dropout(self.dropout)

        input_size = self.pe_dim + self.emb_dim

        self.transformerLayerI = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=8, dropout=self.dropout, dim_feedforward=1024
        )

        self.transformerI = nn.TransformerEncoder(
            self.transformerLayerI,
            num_layers=self.nb_layers,
        )

        self.dropout_m = nn.Dropout(self.dropout)

        if not self.encoder_only:
            # design decoder
            self.transformerLayerO = nn.TransformerDecoderLayer(
                d_model=input_size, nhead=8, dropout=self.dropout, dim_feedforward=1024
            )

            self.transformerO = nn.TransformerDecoder(
                self.transformerLayerO,
                num_layers=self.nb_layers,
            )

            self.dropout_o = nn.Dropout(self.dropout)

        # output layer which projects back to tag space
        self.hidden_to_tag = nn.Linear(self.emb_dim + self.pe_dim, self.nb_tags)

    def __pos_encode(self, p):
        return self.pos_emb[p]

    def forward(self, X, p, X2=None, train_embedding=True):
        self.embedding.weight.requires_grad = train_embedding
        if not self.encoder_only:
            self.embedding2.weight.requires_grad = train_embedding

        I = X

        self.mask = (torch.triu(torch.ones(self.seq_len, self.seq_len)) == 1).transpose(
            0, 1
        )
        self.mask = (
            self.mask.float()
            .masked_fill(self.mask == 0, float("-inf"))
            .masked_fill(self.mask == 1, float(0.0))
        )

        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

        # ---------------------
        # Combine inputs
        X = self.embedding(I)
        X = X.view(self.seq_len, self.batch_size, -1)

        if self.pe_dim > 0:
            P = self.__pos_encode(p)
            P = P.view(self.seq_len, self.batch_size, -1)
            X = torch.cat((X, P), 2)

        X = self.dropout_i(X)

        # Run through transformer encoder

        M = self.transformerI(X, mask=self.mask)
        M = self.dropout_m(M)

        if not self.encoder_only:
            # ---------------------
            # Decoder stack
            X = self.embedding2(X2)
            X = X.view(self.seq_len, self.batch_size, -1)

            if self.pe_dim > 0:
                X = torch.cat((X, P), 2)

            X = self.dropout_i(X)

            X = self.transformerO(X, M, tgt_mask=self.mask, memory_mask=None)
            X = self.dropout_o(X)

            # run through linear layer
            X = self.hidden_to_tag(X)
        else:
            X = self.hidden_to_tag(M)

        Y_hat = X
        return Y_hat
