from torch.utils.data import DataLoader
import wandb

from preprocess import preprocess
from dataset import NotebookDataset
from model import get_model
from trainer import Trainer
from train_utils import yield_optimizer, yield_criterion, yield_scheduler, yield_scaler
from config import Config, WandbConfig


def main():
    # Configuration
    config, wandb_config = Config(), WandbConfig()
    config.mode = "train"

    # Loading Model
    print("Loading Model..: Start")
    tokenizer, model = get_model(config)
    print("Loading Model..: Done!")

    # Loading Data
    print("Loading Data..: Start")
    df_train_md, df_valid_md, df_orders = preprocess(config)
    df_valid_md = df_valid_md[:1000]
    trainset = NotebookDataset(
        df_valid_md, max_len=config.max_len, tokenizer=tokenizer, config=config
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

    logger = wandb.init(
        project="ai4code",
        entity="25111",
        name=config.trial_name,
        config=wandb_config,
        dir=config.log_dir,
    )

    # Train
    trainer = Trainer(
        config,
        dataloaders=[train_loader, valid_loader],
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        scaler=scaler,
        logger=logger
    )

    trainer.train(epochs=config.num_epochs)

    artifact_dataset = wandb.Artifact("ai4code-dataset", type="dataset")
    artifact_dataset.add_file(
        "../input/AI4Code/train.csv", name="input/train.csv"
    )
    artifact_dataset.add_file(
        "../input/AI4Code/train_md.csv", name="input/train_md.csv"
    )
    artifact_dataset.add_file(
        "../input/AI4Code/test.csv", name="input/test.csv"
    )
    artifact_dataset.add_file(
        "../input/AI4Code/test_md.csv", name="input/test_md.csv"
    )
    wandb.run.log_artifact(artifact_dataset)

    artifact_model = wandb.Artifact("ai4code-model", type="model")
    artifact_model.add_dir(
        f"{wandb.run.dir}/{config.trial_name}", name="models"
    )
    wandb.run.log_artifact(artifact_model)

    wandb.run.finish()

if __name__ == "__main__":
    main()
