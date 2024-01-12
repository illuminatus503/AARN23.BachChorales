import torch
import torch.nn as nn

import pytorch_lightning as pl

from .external import VariationalDropout
from .external import CrossEntropyTimeDistributedLoss

from typing import *


class TonicNet(pl.LightningModule):
    def __init__(
        self,
        nb_tags,
        nb_layers=3,
        z_dim=32,
        z_emb_size=32,
        nb_rnn_units=256,
        dropout=0.3,
        batch_size=1,
        seq_len=1,
        base_lr=0.2,
        max_lr=0.2,
    ):
        super(TonicNet, self).__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(
            self.hparams.nb_tags, self.hparams.nb_rnn_units, device=self.device
        )
        self.z_embedding = nn.Embedding(80, self.hparams.z_emb_size, device=self.device)
        self.pos_emb = nn.Embedding(64, 0, device=self.device)  # no es necesario

        self.dropout_i = VariationalDropout(
            max(0.0, self.hparams.dropout - 0.2), batch_first=True
        )

        # INICIALIZAMOS la RNN
        input_size = (
            self.hparams.nb_rnn_units
            + (self.hparams.z_dim > 0) * self.hparams.z_emb_size
        )
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=self.hparams.nb_rnn_units,
            num_layers=self.hparams.nb_layers,
            batch_first=True,
            dropout=self.hparams.dropout,
            device=self.device,
        )

        self.dropout_o = VariationalDropout(self.hparams.dropout, batch_first=True)

        # output layer which projects back to tag space
        self.hidden_to_tag = nn.Linear(
            input_size,
            self.hparams.nb_tags,
            bias=False,
            device=self.device,
        )

        # Loss fn
        self.criterion = CrossEntropyTimeDistributedLoss()

    def init_hidden(self):
        self.hparams.hidden = torch.randn(
            self.hparams.nb_layers,
            self.hparams.batch_size,
            self.hparams.nb_rnn_units,
            device=self.device,
        )

    def forward(
        self,
        X,
        z=None,
        sampling=False,
        reset_hidden=True,
    ):
        # reset the RNN hidden state.
        if not sampling:
            self.hparams.seq_len = X.shape[1]
            if reset_hidden:
                self.init_hidden()

        # ---------------------
        # Combine inputs
        X = self.embedding(X.to(self.device))
        X = X.view(
            self.hparams.batch_size, self.hparams.seq_len, self.hparams.nb_rnn_units
        )

        # Repeating pitch encoding
        if self.hparams.z_dim > 0:
            Z = self.z_embedding(z.to(self.device) % 80)
            Z = Z.view(
                self.hparams.batch_size, self.hparams.seq_len, self.hparams.z_emb_size
            )
            X = torch.cat((Z, X), 2)
        X = self.dropout_i(X)

        # Run through RNN
        X, self.hparams.hidden = self.rnn(X, self.hparams.hidden)
        if self.hparams.z_dim > 0:
            X = torch.cat((Z, X), 2)

        # Run through linear layer
        X = self.dropout_o(X)
        Y_hat = self.hidden_to_tag(X)

        return Y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.base_lr)
        
        # Config. scheduler
        train_dataset_size = len(self.train_dataloader().dataset)
        steps_per_epoch = train_dataset_size // self.hparams.batch_size
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.hparams.max_lr,
            epochs=60,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.8,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=1000.0,
            last_epoch=-1,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        x, y, psx, i, c = batch
        y_hat = self(x, z=i, train_embedding=True)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, psx, i, c = batch
        y_hat = self(x, z=i, train_embedding=False)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5)

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     test_loss = nn.functional.cross_entropy(y_hat, y)
    #     self.log('test_loss', test_loss)

    # def predict_step(self, *args: Any, **kwargs: Any) -> Any:
    #     return super().predict_step(*args, **kwargs)
