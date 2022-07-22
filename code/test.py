import numpy as np
import torch
from config import Config
from dataset import NotebookDataset
from model import get_model
from preprocess import get_features, preprocess
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    config = Config()
    config.mode = "test"

    print("Loading Model..: Start")
    model = get_model(config)
    print("Loading Model..: Done!")

    print("Loading Data..: Start")
    df_test, df_test_md, df_test_py = preprocess(config)

    if config.data_type == "all":
        df_testset = df_test
    elif config.data_type == "md":
        df_testset = df_test_md
    elif config.data_type == "py":
        df_testset = df_test_py
    fts_test = get_features(df_testset)

    testset = NotebookDataset(
        df_testset,
        max_len=config.max_len,
        fts=fts_test,
        config=config,
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
    y_test = test(model, testloader, config)
    print("Testing..: Done!")

    print("Creating submission..: Start")
    if config.data_type == "all":
        df_test["pred"] = y_test
    elif config.data_type == "md":
        df_test.loc[df_test["cell_type"] == "markdown", "pred"] = y_test
    elif config.data_type == "py":
        df_test.loc[df_test["cell_type"] == "code", "pred"] = y_test

    df_submission = (
        df_test.sort_values("pred")
        .groupby("id")["cell_id"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )
    df_submission.rename(columns={"cell_id": "cell_order"}, inplace=True)

    df_submission.to_csv(
        f"output/submission_{str(config.prev_model).split('/')[0]}.csv",
        index=False,
    )
    print("Creating submission..: Done!")


def test(model, dataloader, config):
    model.eval()

    tbar = tqdm(dataloader, total=len(dataloader))

    preds = []
    with torch.no_grad():
        for _, data in enumerate(tbar):
            ids = data[0].to(config.device)
            mask = data[1].to(config.device)
            fts = data[2].to(config.device)

            pred = model(ids=ids, mask=mask, fts=fts).view(-1)

            preds.append(pred.detach().cpu().numpy().ravel())

    return np.concatenate(preds)


if __name__ == "__main__":
    main()
