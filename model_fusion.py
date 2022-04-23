import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics import Accuracy

import argparse

class MNISTFusionModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("MNISTFusionModel")
        # ==== Architecture related ====
        parser.add_argument('--hiddens', type=int, default=512)
        parser.add_argument('--n_layers', type=int, default=3)
        parser.add_argument('--dropout', type=float, default=0.5)

        # ==== Learning related ====
        parser.add_argument('--optim', type=str, default='Adam')
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1e-6)
        parser.add_argument('--temperature', type=float, default=1)
        parser.add_argument('--loss_type', type=str, default="SmoothL1Loss")

        return parent_parser

    def __init__(self, args, pretrained_models):
        super().__init__()
        self.save_hyperparameters(args)

        self.pretrained_models = nn.ModuleList(pretrained_models)
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
        self.loss = getattr(torch.nn, self.hparams.loss_type)()

    def configure_optimizers(self):
        return getattr(torch.optim, self.hparams.optim)(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

    def get_loss_acc(self, x, y, test):
        if test:
            # First get the logits 
            logits = F.log_softmax(self(x), dim=-1)
            loss = F.nll_loss(logits, y)
            self.acc(logits, y)
            return loss
        else:
            # When we are not testing, y is just random
            with torch.no_grad():
                outs = [(F.softmax(model(x), dim=1)).unsqueeze(1) for model in self.pretrained_models]

            # Concatenate and find the entropy 
            temp_distribution = torch.cat(outs, dim=1)

            if self.hparams.temperature > 0:
                entropy = -torch.sum((temp_distribution*torch.log(temp_distribution)), dim=2, keepdim=True)

                # Use softmin of entropy to create weight
                weight = F.softmin(self.hparams.temperature*entropy, dim=1)

                # Weigh the old distribution and pass through loss function
                target = (weight * temp_distribution).sum(dim=1)
            else:
                target = temp_distribution.mean(1)

            return self.loss(F.softmax(self(x), dim=1), target)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.get_loss_acc(x, y, False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.get_loss_acc(x, y, test=True)
        self.log('test_acc', self.acc, on_epoch=True)
        self.log('test_loss', loss, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.get_loss_acc(x, y, True)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', self.acc, on_epoch=True)

    def forward(self, x):
        x = x.flatten(1)
        x = self.in_mlp(x)
        for l in self.layers:
            x = l(x)
        return self.cls(x)
