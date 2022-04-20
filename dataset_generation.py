from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

class MNISTDataModule(pl.LightningDataModule):
    @staticmethod
    def add_mooney_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")

        # ==== Dataloaders config ====
        parser.add_argument("--train_batchsize", type=int, default=128)
        parser.add_argument("--test_batchsize", type=int, default=256)
        parser.add_argument("--val_batchsize", type=int, default=256)
        parser.add_argument("--train_n_workers", type=int, default=32)
        parser.add_argument("--val_n_workers", type=int, default=32)
        parser.add_argument("--test_n_workers", type=int, default=32)

        # ==== Other configs ====

        return parent_parser

    def __init__(self, hparams):
        super().__init__()
        self.data_params = hparams
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=self.data_params.test_batchsize,
            num_workers=self.data_params.test_n_workers)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            shuffle=False,
            batch_size=self.data_params.val_batchsize,
            num_workers=self.data_params.val_n_workers)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.data_params.train_batchsize,
            num_workers=self.data_params.train_n_workers)


