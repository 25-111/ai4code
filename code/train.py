# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2022-06-27 03:31:28
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2022-07-16 04:28:12

import wandb
from config import Config, WandbConfig
from dataset import NotebookDataset
from model import get_model
from preprocess import preprocess
from torch.utils.data import DataLoader
from train_utils import (
    yield_criterion,
    yield_optimizer,
    yield_scaler,
    yield_scheduler,
)
from trainer import get_trainer
import pandas as pd


def main():
    config, wandb_config = Config(), WandbConfig()
    config.mode = "train"

    print("Loading Model..: Start")
    tokenizer, model = get_model(config)
    print("Loading Model..: Done!")

    print("Loading Data..: Start")
    (
        df_train,
        df_valid,
        df_train_md,
        df_valid_md,
        df_train_py,
        df_valid_py,
    ) = preprocess(config)

    # additional data
    df_train_add = pd.read_csv(config.input_dir / "train2.csv").dropna()
    df_train_md = pd.concat([
        df_train_md, df_train_add[df_train_add.cell_type == "markdown"]
    ])
    df_train_md = pd.concat([
        df_train_md, df_train[df_train.cell_type == "code"][:30000]
    ])
    df_train_md = pd.concat([
        df_train_md, df_train_add[df_train_add.cell_type == "code"][:300000]
    ]).drop_duplicates()


    if config.data_type == "all":
        df_train_, df_valid_ = df_train, df_valid
    elif config.data_type == "md":
        df_train_, df_valid_ = df_train_md, df_valid_md
    elif config.data_type == "py":
        df_train_, df_valid_ = df_train_py, df_valid_py
    trainset = NotebookDataset(
        df_train_, max_len=config.max_len, tokenizer=tokenizer, config=config
    )
    validset = NotebookDataset(
        df_valid_, max_len=config.max_len, tokenizer=tokenizer, config=config
    )

    use_pin_mem = config.device.startswith("cuda")
    train_loader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=use_pin_mem,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=use_pin_mem,
    )
    print("Loading Data..: Done!")

    print("Setting hyperparameters..: Done!")
    optimizer = yield_optimizer(model, config)
    criterion = yield_criterion(config)
    scheduler = yield_scheduler(optimizer)
    scaler = yield_scaler()
    print("Setting hyperparameters..: Done!")

    print("Training..: Start")
    run = wandb.init(
        project="ai4code",
        entity="25111",
        name=config.trial_name,
        config=wandb_config,
        dir=config.working_dir,
    )

    trainer = get_trainer(
        config,
        dataloaders=[train_loader, valid_loader],
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        scaler=scaler,
        logger=run,
    )

    trainer.train(epochs=config.num_epochs)
    print("Training..: Done!")

    print("Logging to WandB..: Start")
    artifact_dataset = wandb.Artifact("dataset", type="dataset")
    artifact_dataset.add_file(
        config.input_dir / "train.csv", name="input/train.csv"
    )
    artifact_dataset.add_file(
        config.input_dir / "train_md.csv", name="input/train_md.csv"
    )
    artifact_dataset.add_file(
        config.input_dir / "valid.csv", name="input/valid.csv"
    )
    artifact_dataset.add_file(
        config.input_dir / "valid_md.csv", name="input/valid_md.csv"
    )
    wandb.run.log_artifact(artifact_dataset)

    artifact_model = wandb.Artifact(config.base_model, type="model")
    artifact_model.add_dir(
        config.working_dir / config.base_model / config.trial_name,
        name=f"{config.trial_name}",
    )
    wandb.run.log_artifact(artifact_model)

    wandb.run.finish()
    print("Logging to WandB..: Done!")


if __name__ == "__main__":
    main()
