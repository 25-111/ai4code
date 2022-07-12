import argparse
from os import path as osp

import numpy as np
import torch
from config import Config
from dataset import NotebookDataset
from model import get_model
from preprocess import get_features, preprocess
from torch.utils.data import DataLoader
from tqdm import tqdm

MODEL_PATH = "220711-0122-microsoft/codebert-base-AdamW-MSE/model_0.pth"


def main(args):
    config = Config()
    config.mode = "test"

    # Loading Model
    print("Loading Model..: Start")
    tokenizer, model = get_model(config)
    print("Loading Model..: Done!")

    model.load_state_dict(torch.load(osp.join(Config.working_dir, args.model_path)))
    df_test, df_test_md = preprocess(config)

    print("Loading Data..: Start")
    testset = NotebookDataset(
        df_test_md, max_len=config.max_len, tokenizer=tokenizer, config=config
    )
    testloader = DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    print("Loading Data..: Done!")

    print("Testing..: Start!")
    _, y_test = test(model, testloader, config)
    print("Testing..: Done!")

    print("Creating submission..: Start!")
    df_test.loc[df_test["cell_type"] == "markdown", "pred"] = y_test

    df_submission = (
        df_test.sort_values("pred")
        .groupby("id")["cell_id"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )
    df_submission.rename(columns={"cell_id": "cell_order"}, inplace=True)

    df_submission.to_csv(
        f"submission_{config.timestamp}.csv",
        index=False,
    )
    print("Creating submission..: Done!")


def test(model, dataloader, config):
    model.eval()

    tbar = tqdm(dataloader, total=len(dataloader))

    preds, labels = [], []
    with torch.no_grad():
        for _, data in enumerate(tbar):
            ids = data[0].to(config.device)
            mask = data[1].to(config.device)
            ttis = data[2].to(config.device)
            targets = data[-1].to(config.device)

            pred = model(ids=ids, mask=mask, token_type_ids=ttis).view(-1)

            labels.append(targets.detach().cpu().numpy().ravel())
            preds.append(pred.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-mp", type=str, default=MODEL_PATH)
    args = parser.parse_args()

    main(args)
