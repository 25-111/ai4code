import sys
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from preprocess import preprocess, get_features
from dataset import NotebookDataset, read_data
from model import get_model
from config import Config


def main():
    config = Config()
    config.mode = "test"

    # Loading Model
    print("Loading Model..: Start")
    tokenizer, model = get_model(config)
    print("Loading Model..: Done!")

    model.load_state_dict(torch.load("../working/220711-0122-microsoft/codebert-base-AdamW-MSE/model_0.pth"))
    df_test, df_test_md = preprocess(config)

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

    _, y_test = test(model, testloader, config)
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
    main()
