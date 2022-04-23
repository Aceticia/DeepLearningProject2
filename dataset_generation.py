import torch
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from utils import get_dataset

class MNISTDataModule(pl.LightningDataModule):
    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")

        # ==== Dataloaders config ====
        parser.add_argument("--train_batchsize", type=int, default=256)
        parser.add_argument("--test_batchsize", type=int, default=512)
        parser.add_argument("--val_batchsize", type=int, default=512)
        parser.add_argument("--train_n_workers", type=int, default=32)
        parser.add_argument("--val_n_workers", type=int, default=32)
        parser.add_argument("--test_n_workers", type=int, default=32)

        # ==== File configs ====
        parser.add_argument("--root", type=str, default="~/data/")

        # ==== Other configs ====
        parser.add_argument("--partition_rnd_state", type=int, default=42)
        parser.add_argument("--partition_num", type=int, default=0)
        parser.add_argument("--val_ratio", type=float, default=0.2)

        return parent_parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # Get the mnist dataset
        train_val_dataset = get_dataset(
            idx=self.hparams.partition_num,
            root=self.hparams.root,
            download=True,
            train=True,
        )
        test_dataset = get_dataset(
            idx=self.hparams.partition_num,
            root=self.hparams.root,
            download=True,
            train=False,
        )

        # Further divide the train_val partition into train and val
        train_val_length = [int(len(train_val_dataset)*(1-self.hparams.val_ratio))]
        train_val_length += [len(train_val_dataset)-train_val_length[0]]
        train_dataset, val_dataset = random_split(
            train_val_dataset, train_val_length, generator=torch.Generator().manual_seed(self.hparams.partition_rnd_state))

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=self.hparams.test_batchsize,
            num_workers=self.hparams.test_n_workers)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            shuffle=False,
            batch_size=self.hparams.val_batchsize,
            num_workers=self.hparams.val_n_workers)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.train_batchsize,
            num_workers=self.hparams.train_n_workers)


