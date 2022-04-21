from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from argparse import ArgumentParser

from dataset_generation import MNISTDataModule as DataModule
from model import MNISTModel as Model

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = DataModule.add_dataset_specific_args(parser)
    parser = Model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--wandb_run_name', type=str, default="debug")
    parser.add_argument('--wandb_project_name', type=str, default="debug")
    parser.add_argument('--partition_ckpt_directory', type=str, default="./split30")
    args = parser.parse_args()

    # Instantiate the dataset
    datamodule = DataModule(args)

    # Instantiate model
    model = Model(args)

    # Instantiate wandb logger
    logger = WandbLogger(
        name=args.wandb_run_name,
        project=args.wandb_project_name
    )

    # Instantiate trainer
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                monitor='val_loss',
                dirpath=args.partition_ckpt_directory,
                filename=args.wandb_run_name
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=50
            )
        ]
    )

    # Train
    trainer.fit(
        model=model,
        datamodule=datamodule
    )

    # Find best checkpoint to test
    trainer.test(
        ckpt_path=trainer.checkpoint_callback.best_model_path,
        datamodule=datamodule
    )

