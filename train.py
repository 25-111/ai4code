import sys
import argparse
from tqdm import tqdm
import wandb
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from data import preprocess, get_loaders
from model import get_model
from train import train
from config import Config, WandbConfig
from utils import adjust_lr, calc_kendall_tau, wandb_log


def main(config):
    # Configuration
    config = Config()
    config.mode = "train"

    if config.wandb_key:
        wandb_config = WandbConfig()
        wandb.login(key=config.wandb_key)

        run = wandb.init(project="ai4code", config=wandb_config, entity="nciaproject")

    # Loading Data
    df_orders, _, df_valid, df_train_md, df_valid_md = preprocess(config)
    trainloader, validloader = get_loaders(df_train_md, df_valid_md, config)

    # Loading Model
    model = get_model(config).to(config.device)

    # Setting Train
    optimizer = yield_optim(model, config)
    criterion = yield_loss(config)

    # Train
    model, y_pred = train(model, trainloader, validloader, optimizer, criterion, config)

    # Evaluate Perforemance
    df_valid["pred"] = df_valid.groupby(["id", "cell_type"])["rank"].rank(pct=True)
    df_valid.loc[df_valid["cell_type"] == "markdown", "pred"] = y_pred
    y_dummy = df_valid.sort_values("pred").groupby("id")["cell_id"].apply(list)
    print("Kendall_tau: ", calc_kendall_tau(df_orders.loc[y_dummy.index], y_dummy))

    if config.wandb_key:
        run.finish()


def yield_optim(model, config):
    if config.optim == "Adam":
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr
        )
    elif config.optim == "AdamW":
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr
        )


def yield_loss(config):
    if config.loss == "MSE":
        torch.nn.MSELoss()


def validate(model, validloader, config):
    model.eval()

    tbar = tqdm(validloader, file=sys.stdout)

    preds, labels = [], []
    with torch.no_grad():
        for _, data in enumerate(tbar):
            inputs, label = (d.to(config.device) for d in data[:-1]), data[-1].to(
                config.device
            )

            pred = model(inputs[0], inputs[1])

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(label.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)


def train(model, trainloader, validloader, optimizer, criterion, config):
    np.random.seed(0)

    for epoch in range(config.num_epochs):
        model.train()
        tbar = tqdm(trainloader, file=sys.stdout)

        lr = adjust_lr(optimizer, epoch)

        losses, preds, labels = [], [], []
        for _, data in enumerate(tbar):
            inputs, label = (d.to(config.device) for d in data[:-1]), data[-1].to(
                config.device
            )

            optimizer.zero_grad()
            pred = model(inputs[0], inputs[1])

            loss = criterion(pred, label)
            wandb_log(train_loss=loss.item())

            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(label.detach().cpu().numpy().ravel())

            tbar.set_description(
                f"Epoch {epoch + 1} Loss: {np.mean(losses):-4e} lr: {lr}"
            )

        y_val, y_pred = validate(model, validloader)

        print("Validation MSE:", np.round(mean_squared_error(y_val, y_pred), 4))
        print()
    return model, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="name of model")
    args = parser.parse_args()

    main(args)
