import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from TonicNet.audio.dataset import BachChoralesDataset
from TonicNet.models.utils import print_model_summary

from TonicNet.models.external import (
    VariationalDropout,
    CrossEntropyTimeDistributedLoss,
    LiGRU,
)

from typing import *


class TonicNet(pl.LightningModule):
    def __init__(
        self,
        train_dir,
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

        self.embedding = nn.Embedding(nb_tags, nb_rnn_units, device=self.device)
        self.z_embedding = nn.Embedding(80, z_emb_size, device=self.device)
        self.pos_emb = nn.Embedding(64, 0, device=self.device)  # no es necesario

        self.dropout_i = VariationalDropout(max(0.0, dropout - 0.2), batch_first=True)

        # INICIALIZAMOS la RNN
        input_size = nb_rnn_units + (z_dim > 0) * z_emb_size
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=nb_rnn_units,
            num_layers=nb_layers,
            batch_first=True,
            dropout=dropout,
            device=self.device,
        )
        # self.rnn = LiGRU(
        #     input_size,
        #     hidden_size=nb_rnn_units,
        #     num_layers=nb_layers,
        #     dropout=dropout,
        # )

        # Define output layer which projects back to tag space
        self.dropout_o = VariationalDropout(dropout, batch_first=True)
        self.hidden_to_tag = nn.Linear(
            input_size, nb_tags, bias=False, device=self.device
        )

        # Set loss fn
        self.criterion = CrossEntropyTimeDistributedLoss()

        # Train dataset:
        # Create the dataset
        self.train_dataset = BachChoralesDataset(
            self.hparams.train_dir,
            device=self.device,
            lazy=True,
        )

        # Set the batch_size to a single batch
        self.hparams.batch_size = len(self.train_dataset) + 76
        self.step_size = 3 * self.hparams.batch_size

    def init_hidden(self):
        self.hparams.hidden = torch.randn(
            self.hparams.nb_layers,
            self.hparams.batch_size,
            self.hparams.nb_rnn_units,
            device=self.device,
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False, # Error!! No shuffle
            num_workers=4,
            persistent_workers=True,
        )
        return train_loader

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
        X = self.embedding(X.to(self.device).long())
        X = X.view(-1, self.hparams.seq_len, self.hparams.nb_rnn_units)

        # Repeating pitch encoding
        if z is not None and self.hparams.z_dim > 0:
            Z = self.z_embedding(z.to(self.device) % 80)
            Z = Z.view(-1, self.hparams.seq_len, self.hparams.z_emb_size)
            X = torch.cat((Z, X), 2)
        X = self.dropout_i(X)

        # Run through RNN
        X, self.hparams.hidden = self.rnn(X, self.hparams.hidden)
        if z is not None and self.hparams.z_dim > 0:
            X = torch.cat((Z, X), 2)

        # Run through linear layer
        X = self.dropout_o(X)
        Y_hat = self.hidden_to_tag(X)

        return Y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.base_lr)

        # Config. scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.hparams.max_lr,
            epochs=60,
            steps_per_epoch=self.step_size,
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
        x, y, _, i, _ = batch
        y_hat = self(x, z=i)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y, psx, i, c = batch
    #     y_hat = self(x, z=i, train_embedding=False)
    #     loss = self.criterion(y_hat, y)
    #     self.log("val_loss", loss)
    #     return loss

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     test_loss = nn.functional.cross_entropy(y_hat, y)
    #     self.log('test_loss', test_loss)

    # def predict_step(self, *args: Any, **kwargs: Any) -> Any:
    #     return super().predict_step(*args, **kwargs)

    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5)

    def summary(self):
        print_model_summary(self)
