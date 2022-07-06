import sys, warnings
import wandb
import numpy as np
import torch
import transformers as tx
from preprocess import preprocess, get_features

# TODO: read_data 처리 (아마 Dataset에서 처리하는 것이 나을 듯)
from dataset import NotebookDataset, read_data
from model import get_model
from config import Config, WandbConfig
from torch.utils.data import DataLoader
from trainer import Trainer
from torch import optim
from torch.nn import MSELoss
from torch.cuda.amp import GradScaler


def main():
    # Configuration
    config = Config()
    config.mode = "train"


    # Loading Model
    print("Loading Model..: Start")
    tokenizer, model = get_model(config)
    print("Loading Model..: Done!")
    # Loading Data
    use_pin_mem = config.device.startswith("cuda")

    print("Loading Data..: Start")
    df_train, df_train_md, df_valid, df_valid_md, df_orders = (
        preprocess(config)
    )
    fts_train, fts_valid = get_features(df_train), get_features(df_valid)

    trainset = NotebookDataset(
        df_train_md,
        max_len=config.max_len,
        max_len_md=config.max_len_md,
        fts=fts_train,
        tokenizer=tokenizer,
    )
    validset = NotebookDataset(
        df_valid_md,
        max_len=config.max_len,
        max_len_md=config.max_len_md,
        fts=fts_valid,
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        validset, batch_size=config.batch_size, shuffle=False
    )
    print("Loading Data..: Done!")
    
    print("Setting hyperparameters..: Done!")
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    loss_fn = MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=50,
        eta_min=3e-5
    )
    print("Setting hyperparameters..: Done!")

    # Train
    trainer = Trainer(
        config,
        dataloaders=[train_loader, valid_loader],
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        scheduler=scheduler,
        scaler=GradScaler(),
        device="cuda"
    )

    trainer.train(epochs=config.num_epochs)


if __name__ == "__main__":
    main()
