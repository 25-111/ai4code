import sys
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from data import get_loaders #! remove for real train
from model import MarkdownModel #! remove for real train
from utils import adjust_lr #! remove for real train
from config import TrainConfig #! remove for real train

trainloader, validloader = get_loaders(TrainConfig)

model = MarkdownModel(ModelConfig)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4,
    betas=(0.9, 0.999),
    eps=1e-08,
)


def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validate(model, validloader):
    model.eval()

    tbar = tqdm(validloader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for _, data in enumerate(tbar):
            inputs, target = read_data(data)

            pred = model(inputs[0], inputs[1])

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)


def train(model, trainloader, validloader, optimizer, criterion, epochs):
    np.random.seed(0)

    for e in range(epochs):
        model.train()
        tbar = tqdm(trainloader, file=sys.stdout)

        lr = adjust_lr(optimizer, e)

        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            optimizer.zero_grad()
            pred = model(inputs[0], inputs[1])

            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e+1} Loss: {avg_loss} lr: {lr}")

        y_val, y_pred = validate(model, validloader)

        print("Validation MSE:", np.round(mean_squared_error(y_val, y_pred), 4))
        print()
    return model, y_pred
