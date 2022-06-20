from datetime import datetime
from pytz import timezone
from torch.utils.data import DataLoader
from data import preprocess, NotebookDataset
from model import get_model
from train import validate
from config import Config


def test():
    config = Config()
    config.mode = "test"

    df, df_test_md = preprocess(config)

    testset = NotebookDataset(
        df_test_md,
        max_len=config.max_len,
    )
    testloader = DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )

    model = get_model(config)

    _, y_test = validate(model, testloader)
    df.loc[df["cell_type"] == "markdown", "pred"] = y_test

    sub_df = (
        df.sort_values("pred")
        .groupby("id")["cell_id"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )
    sub_df.rename(columns={"cell_id": "cell_order"}, inplace=True)

    time_stamp = datetime.now(timezone("Asia/Seoul")).strftime("%y%m%d-%H%M")
    sub_df.to_csv(
        f"./results/submission_{config.model_name}_{time_stamp}.csv", index=False
    )


if __name__ == "__main__":
    test()
