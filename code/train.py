from torch.utils.data import DataLoader

# TODO: read_data 처리 (아마 Dataset에서 처리하는 것이 나을 듯)
from preprocess import preprocess
from dataset import NotebookDataset
from model import get_model
from trainer import Trainer
from train_utils import yield_optimizer, yield_criterion, yield_scheduler, yield_scaler
from config import Config


def main():
    # Configuration
    config = Config()
    config.mode = "train"

    # Loading Model
    print("Loading Model..: Start")
    tokenizer, model = get_model(config)
    print("Loading Model..: Done!")

    # Loading Data
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

    # Train
    trainer = Trainer(
        config,
        dataloaders=[train_loader, valid_loader],
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        scaler=scaler,
    )

    trainer.train(epochs=config.num_epochs)


if __name__ == "__main__":
    main()
