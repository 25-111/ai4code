import numpy as np
import torch
from config import Config
from dataset import NotebookDataset
from model import get_model
from preprocess import preprocess
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    config = Config()
    config.mode = "test"

    print("Loading Model..: Start")
    tokenizer, model = get_model(config)
    print("Loading Model..: Done!")

    print("Loading Data..: Start")
    df_test, df_test_md, df_test_py = preprocess(config)

    if config.data_type == "all":
        df_test_ = df_test
    elif config.data_type == "md":
        df_test_ = df_test_md
    elif config.data_type == "py":
        df_test_ = df_test_py
    testset = NotebookDataset(
        df_test_, max_len=config.max_len, tokenizer=tokenizer, config=config
    )
    testloader = DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    print("Loading Data..: Done!")

    print("Testing..: Start")
    _, y_test = test(model, testloader, config)
    print("Testing..: Done!")

    print("Creating submission..: Start")
    df_submission_ = df_test.copy()
    df_submission_.loc[
        df_submission_["cell_type"] == "markdown", "pred"
    ] = y_test

    df_submission = (
        df_submission_.sort_values("pred")
        .groupby("id")["cell_id"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )
    df_submission.rename(columns={"cell_id": "cell_order"}, inplace=True)

    df_submission.to_csv(
        f"output/submission_{config.timestamp}.csv", index=False
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
    main()
