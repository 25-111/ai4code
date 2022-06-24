from datetime import datetime
from pytz import timezone
from torch.utils.data import DataLoader
from preprocess import preprocess, get_features
from dataset import preprocess, NotebookDataset
from model import get_model
from train import validate
from config import Config


def test():
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
    testloader = DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )

    _, y_test = validate(model, testloader)
    df_test.loc[df_test["cell_type"] == "markdown", "pred"] = y_test

    sub_df = (
        df_test.sort_values("pred")
        .groupby("id")["cell_id"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )
    sub_df.rename(columns={"cell_id": "cell_order"}, inplace=True)

    time_stamp = datetime.now(timezone("Asia/Seoul")).strftime("%y%m%d-%H%M")
    sub_df.to_csv(
        f"{config.result_dir}/submission_{config.model_name}_{time_stamp}.csv",
        index=False,
    )


if __name__ == "__main__":
    test()
