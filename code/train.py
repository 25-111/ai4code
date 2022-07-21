import wandb
from config import Config, WandbConfig
from dataset import NotebookDataset
from model import get_model
from preprocess import get_features, preprocess
from torch.utils.data import DataLoader
from train_utils import (
    yield_criterion,
    yield_optimizer,
    yield_scaler,
    yield_scheduler,
)
from trainer import Trainer


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

    if config.data_type == "all":
        df_trainset, df_validset = df_train, df_valid
    elif config.data_type == "md":
        df_trainset, df_validset = df_train_md, df_valid_md
    elif config.data_type == "py":
        df_trainset, df_validset = df_train_py, df_valid_py
    fts_train, fts_valid = get_features(df_trainset), get_features(df_validset)

    trainset = NotebookDataset(
        df_trainset,
        max_len=config.max_len,
        tokenizer=tokenizer,
        fts=fts_train,
        config=config,
    )
    validset = NotebookDataset(
        df_validset,
        max_len=config.max_len,
        tokenizer=tokenizer,
        fts=fts_valid,
        config=config,
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
        dir=config.working_dir,
        config=wandb_config,
        project="ai4code",
        entity="25111",
        name=config.trial_name,
    )

    trainer = Trainer(
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
