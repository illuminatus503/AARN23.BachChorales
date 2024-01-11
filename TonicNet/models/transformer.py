import numpy as np

import torch
import torch.nn as nn

from ..audio import MAX_SEQ


def position_encoding_init(n_position: int, emb_dim: int) -> torch.Tensor:
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
        requires_grad=False,
    )

    position_enc[1:, 0::2] = np.sin(
        position_enc[1:, 0::2]
    )  # apply sin on 0th,2nd,4th...emb_dim

    position_enc[1:, 1::2] = np.cos(
        position_enc[1:, 1::2]
    )  # apply cos on 1st,3rd,5th...emb_dim

    return position_enc


class TransformerEmbedding(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, seq_len, batch_size, pos_embedding_dim=0
    ):
        super(TransformerEmbedding, self).__init__()
        
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._seq_len = seq_len
        self._batch_size = batch_size

        if pos_embedding_dim > 0:
            self._pos_embedding = position_encoding_init(MAX_SEQ, pos_embedding_dim)
        else:
            self._pos_embedding = None

    def forward(self, input, p=None):
        x = self._embedding(input)
        # x = x.view(self._seq_len, self._batch_size, -1).contiguous()

        if p is not None and self._pos_embedding:
            p = self._pos_embedding(p)
            # p = p.view(self._seq_len, self._batch_size, -1).contiguous()
            # x = torch.cat((x, p), 2)
            x += p
            print(x.shape)

        return x


class TransformerModel(nn.Module):
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
        super(TransformerModel, self).__init__()

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
        # MÁSCARAS PARA EL APRENDIZAJE CAUSAL
        self.mask = torch.triu(torch.ones(self.seq_len, self.seq_len)).T
        self.mask = self.mask.masked_fill(self.mask == 0, -torch.inf)
        self.mask = self.mask.masked_fill(self.mask == 1, 0.0)

        # ENCODER
        self.embedding_encoder = TransformerEmbedding(
            self.nb_tags,
            self.emb_dim,
            self.seq_len,
            self.batch_size,
        )
        self.dropout_i = nn.Dropout(self.dropout)

        self.transformerLayerI = nn.TransformerEncoderLayer(
            d_model=self.pe_dim + self.emb_dim,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=1024,
        )

        self.transformerI = nn.TransformerEncoder(
            self.transformerLayerI,
            num_layers=self.nb_layers,
        )

        self.dropout_m = nn.Dropout(self.dropout)

        if not self.encoder_only:
            # DECODER
            self.embedding_decoder = TransformerEmbedding(
                self.nb_tags,
                self.emb_dim,
                self.seq_len,
                self.batch_size,
            )

            self.transformerLayerO = nn.TransformerDecoderLayer(
                d_model=self.pe_dim + self.emb_dim,
                nhead=8,
                dropout=self.dropout,
                dim_feedforward=1024,
            )

            self.transformerO = nn.TransformerDecoder(
                self.transformerLayerO,
                num_layers=self.nb_layers,
            )

            self.dropout_o = nn.Dropout(self.dropout)

        # output layer which projects back to tag space
        self.hidden_to_tag = nn.Linear(self.pe_dim + self.emb_dim, self.nb_tags)

    def forward(self, X, p, X2=None, train_embedding=True):
        # self.embedding_encoder.weight.requires_grad = train_embedding
        # if not self.encoder_only:
        #     self.embedding_decoder.weight.requires_grad = train_embedding

        # ---------------------
        # Combine inputs & run through transformer encoder
        X = self.embedding_encoder(X, p)
        X = self.dropout_i(X)
        M = self.transformerI(X, self.mask)
        M = self.dropout_m(M)

        if not self.encoder_only:
            # ---------------------
            # Decoder stack
            X2 = self.embedding_decoder(X2)
            X2 = self.dropout_i(X2)
            X2 = self.transformerO(X2, M, tgt_mask=self.mask, memory_mask=None)
            M = self.dropout_o(X2)

        # run through linear layer
        Y_hat = self.hidden_to_tag(M)

        return Y_hat
