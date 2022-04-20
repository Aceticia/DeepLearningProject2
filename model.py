import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics import Accuracy

import argparse

class MNISTModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("MNISTModel")
        # ==== Architecture related ====
        parser.add_argument('--hiddens', type=int, default=512)
        parser.add_argument('--n_layers', type=int, default=3)
        parser.add_argument('--dropout', type=float, default=0.5)

        # ==== Learning related ====
        parser.add_argument('--optim', type=str, default='Adam')
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1e-6)
        return parent_parser

    def __init__(self, args):
        super(MNISTModel).__init__()
        self.save_hyperparameters()

        self.layers = nn.ModuleList()
        self.in_mlp = nn.Sequential(
            nn.Linear(28*28, self.hparams.hiddens),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout)
        )
        for _ in range(self.hparams.n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(self.hparams.hiddens, self.hparams.hiddens),
                    nn.ReLU(),
                    nn.Dropout(self.hparams.dropout)
                )
            )
        self.cls = nn.Linear(self.hparams.hiddens, 10)
        self.acc = Accuracy()

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.hparams.optim)(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optim

    def get_loss_acc(self, x, y):
        # First get the logits 
        logits = F.log_softmax(self(x), dim=-1)
        loss = F.nll_loss(logits, y)
        self.acc(logits, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.get_loss_acc(x, y)
        self.log('train_acc', self.acc, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.get_loss_acc(x, y)
        self.log('test_acc', self.acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.get_loss_acc(x, y)
        self.log('val_acc', self.acc, on_epoch=True)
        return loss

    def forward(self, x):
        x = x.flatten(1)
        for l in self.layers:
            x = l(x)
        return self.cls(x)
