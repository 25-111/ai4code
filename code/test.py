import sys
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data as dt
from preprocess import preprocess, get_features
from dataset import NotebookDataset, read_data
from model import get_model
from config import Config


def main():
    config = Config()
    config.mode = "test"

    tokenizer, model = get_model(config)

    df_test, df_test_md = preprocess(config)
    fts_test = get_features(df_test)

    testset = NotebookDataset(
        df_test_md,
        max_len=config.max_len,
        max_len_md=config.max_len_md,
        fts=fts_test,
        tokenizer=tokenizer,
    )
    testloader = dt.DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )

    _, y_test = test(model, testloader)
    df_test.loc[df_test["cell_type"] == "markdown", "pred"] = y_test

    sub_df = (
        df_test.sort_values("pred")
        .groupby("id")["cell_id"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )
    sub_df.rename(columns={"cell_id": "cell_order"}, inplace=True)

    sub_df.to_csv(
        f"{config.result_dir}/submission_{config.model_name}_{config.timestamp}.csv",
        index=False,
    )


def test(model, dataloader, config):
    model.eval()

    tbar = tqdm(dataloader, file=sys.stdout)

    preds, labels = [], []
    with torch.no_grad():
        for _, data in enumerate(tbar):
            inputs, labels = read_data(data, config)

            pred = model(*inputs)

            labels.append(labels.detach().cpu().numpy().ravel())
            preds.append(pred.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)


if __name__ == "__main__":
    main()
