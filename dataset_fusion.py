import torch
from torchvision.datasets import MNIST
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


class RandDataset(Dataset):
    def __init__(self, length, size, mean, std) -> None:
        super().__init__()
        self.length = length
        self.size = size
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        # Label is bogus
        return torch.randn(self.size), 1

class MNISTFusionDataModule(pl.LightningDataModule):
    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")

        # ==== Dataloaders config ====
        parser.add_argument("--train_batchsize", type=int, default=128)
        parser.add_argument("--test_batchsize", type=int, default=256)
        parser.add_argument("--val_batchsize", type=int, default=256)
        parser.add_argument("--train_n_workers", type=int, default=32)
        parser.add_argument("--val_n_workers", type=int, default=32)
        parser.add_argument("--test_n_workers", type=int, default=32)

        # ==== File configs ====
        parser.add_argument("--root", type=str, default="~/data/")
        parser.add_argument("--dataset_length", type=int, default=512)
        return parent_parser

    def __init__(self, hparams):
        super().__init__(hparams)
        self.save_hyperparameters(hparams)

        # Get the mnist dataset
        transform = T.Compose([
            T.ToTensor(), 
            T.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = MNIST(
            root=self.hparams.root,
            download=True,
            train=False,
            transform=transform,
        )
        self.train_dataset = RandDataset(
            length=self.hparams.dataset_length,
            size=(28, 28),
            mean=0.1307,
            std=0.3081
        )
        self.val_dataset = RandDataset(
            length=self.hparams.dataset_length,
            size=(28, 28),
            mean=0.1307,
            std=0.3081
        )
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


