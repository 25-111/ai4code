from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils.data import DataLoader, Dataset
from utils import read_notebook, get_ranks


class NotebookDataset(Dataset):
    def __init__(self, df, max_len):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_len = max_len

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        ids = torch.LongTensor(inputs["input_ids"])
        mask = torch.LongTensor(inputs["attention_mask"])

        return ids, mask, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.df.shape[0]


def preprocess(config):
    data_pth = list((config.data_dir / config.mode).glob("*.json"))
    notebooks = [read_notebook(path) for path in tqdm(data_pth, desc="Reading NBs")]

    if config.mode == "train":
        df = (
            pd.concat(notebooks)
            .set_index("id", append=True)
            .swaplevel()
            .sort_index(level="id", sort_remaining=False)
        )

        df_orders = pd.read_csv(
            config.data_dir / "train_orders.csv",
            index_col="id",
            squeeze=True,
        ).str.split()  # Split the string representation of cell_ids into a list

        df_orders_ = df_orders.to_frame().join(
            df.reset_index("cell_id").groupby("id")["cell_id"].apply(list),
            how="right",
        )
        ranks = {
            id_: {"cell_id": cell_id, "rank": get_ranks(cell_order, cell_id)}
            for id_, cell_order, cell_id in df_orders_.itertuples()
        }

        df_ranks = (
            pd.DataFrame.from_dict(ranks, orient="index")
            .rename_axis("id")
            .apply(pd.Series.explode)
            .set_index("cell_id", append=True)
        )

        df_ancestors = pd.read_csv(
            config.data_dir / "train_ancestors.csv", index_col="id"
        )

        df = (
            df.reset_index()
            .merge(df_ranks, on=["id", "cell_id"])
            .merge(df_ancestors, on=["id"])
        )

        df["pct_rank"] = df["rank"] / df.groupby("id")["cell_id"].transform("count")

        splitter = GroupShuffleSplit(
            n_splits=1, test_size=config.valid_ratio, random_state=0
        )
        idx_train, idx_valid = next(splitter.split(df, groups=df["ancestor_id"]))
        df_train = df.loc[idx_train].reset_index(drop=True)
        df_valid = df.loc[idx_valid].reset_index(drop=True)

        df_train_md = df_train[df_train["cell_type"] == "markdown"].reset_index(
            drop=True
        )
        df_valid_md = df_valid[df_valid["cell_type"] == "markdown"].reset_index(
            drop=True
        )

        return df_orders, df_train, df_valid, df_train_md, df_valid_md

    elif config.mode == "test":
        df = (
            pd.concat(notebooks)
            .set_index("id", append=True)
            .swaplevel()
            .sort_index(level="id", sort_remaining=False)
        ).reset_index()

        df["rank"] = df.groupby(["id", "cell_type"]).cumcount()
        df["pred"] = df.groupby(["id", "cell_type"])["rank"].rank(pct=True)

        df["pct_rank"] = 0

        df_test_md = df[df["cell_type"] == "markdown"].reset_index(drop=True)

        return df, df_test_md


def get_loaders(df_train_md, df_valid_md, config):
    use_pin_mem = config.device.startswith("cuda")

    trainset = NotebookDataset(df_train_md, max_len=config.max_len)
    validset = NotebookDataset(df_valid_md, max_len=config.max_len)

    trainloader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_pin_mem,
        drop_last=True,
    )
    validloader = DataLoader(
        validset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_mem,
        drop_last=False,
    )
    return trainloader, validloader
