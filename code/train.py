import wandb
from config import Config, WandbConfig
from dataset import NotebookDataset
from model import get_model
from preprocess import preprocess
from torch.utils.data import DataLoader
from train_utils import yield_criterion, yield_optimizer, yield_scaler, yield_scheduler
from trainer import Trainer


def main():
    config, wandb_config = Config(), WandbConfig()
    config.mode = "train"

    print("Loading Model..: Start")
    tokenizer, model = get_model(config)
    print("Loading Model..: Done!")

    print("Loading Data..: Start")
    df_train_md, df_valid_md, df_orders = preprocess(config)
    trainset = NotebookDataset(
        df_train_md, max_len=config.max_len, tokenizer=tokenizer, config=config
    )
    validset = NotebookDataset(
        df_valid_md, max_len=config.max_len, tokenizer=tokenizer, config=config
    )

    use_pin_mem = config.device.startswith("cuda")
    train_loader = DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, pin_memory=use_pin_mem
    )
    valid_loader = DataLoader(
        validset, batch_size=config.batch_size, shuffle=False, pin_memory=use_pin_mem
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
    # no upload unless the dataset changes
    artifact_dataset = wandb.Artifact("dataset", type="dataset")
    artifact_dataset.add_file(config.input_dir / "train.csv", name="input/train.csv")
    artifact_dataset.add_file(
        config.input_dir / "train_md.csv", name="input/train_md.csv"
    )
    artifact_dataset.add_file(config.input_dir / "valid.csv", name="input/valid.csv")
    artifact_dataset.add_file(
        config.input_dir / "valid_md.csv", name="input/valid_md.csv"
    )
    wandb.run.log_artifact(artifact_dataset)

    # TBD: Should we upload all checkpoint models?
    artifact_model = wandb.Artifact("model", type="model")
    artifact_model.add_dir(
        config.working_dir / "models" / config.trial_name,
        name=f"models/{config.trial_name}",
    )
    wandb.run.log_artifact(artifact_model)

    wandb.run.finish()
    print("Logging to WandB..: Done!")


if __name__ == "__main__":
    main()
