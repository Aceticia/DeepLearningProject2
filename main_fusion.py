import os

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
)

from argparse import ArgumentParser

from dataset_fusion import MNISTFusionDataModule as DataModule
from model import MNISTModel as PretrainedModel
from model_fusion import MNISTFusionModel as FusionModel

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = DataModule.add_dataset_specific_args(parser)
    parser = FusionModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_project_name', type=str, default=None)
    parser.add_argument('--partition_ckpt_directory', type=str, default="./split2")
    parser.add_argument('--fusion_outcome_ckpt_directory', type=str, default="./split2_fusion_outcome")
    args = parser.parse_args()

    # Instantiate the dataset
    datamodule = DataModule(args)

    # Instantiate pretrained models
    pretrained_models = []
    for f in os.listdir(args.partition_ckpt_directory):
        full_path = os.path.join(args.partition_ckpt_directory, f)
        pretrained_models.append(PretrainedModel.load_from_checkpoint(full_path))

    # Instantiate fusion model
    model = FusionModel(args, pretrained_models=pretrained_models)

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
                dirpath=args.fusion_outcome_ckpt_directory,
                filename=args.wandb_run_name
            ),
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

