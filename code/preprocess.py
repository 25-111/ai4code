import json
from os import path as osp

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm


def preprocess(config):
    if not osp.exists(config.input_dir / f"{config.mode}.csv"):
        data_path = list((config.input_dir / config.mode).glob("*.json"))
        notebooks = [
            read_notebook(path)
            for path in tqdm(data_path, desc="Reading notebooks")
        ]

        df = (
            pd.concat(notebooks)
            .set_index("id", append=True)
            .swaplevel()
            .sort_index(level="id", sort_remaining=False)
        )

        if config.mode == "train":
            df_orders = pd.read_csv(
                config.input_dir / "train_orders.csv",
                index_col="id",
                squeeze=True,
            ).str.split()

            df_orders_ = df_orders.to_frame().join(
                df.reset_index("cell_id").groupby("id")["cell_id"].apply(list),
                how="right",
            )

            ranks = {
                id_: {
                    "cell_id": cell_id,
                    "rank": get_ranks(cell_order, cell_id),
                }
                for id_, cell_order, cell_id in df_orders_.itertuples()
            }
            df_ranks = (
                pd.DataFrame.from_dict(ranks, orient="index")
                .rename_axis("id")
                .apply(pd.Series.explode)
                .set_index("cell_id", append=True)
            )

            df_ancestors = pd.read_csv(
                config.input_dir / "train_ancestors.csv", index_col="id"
            )
            df = (
                df.reset_index()
                .merge(df_ranks, on=["id", "cell_id"])
                .merge(df_ancestors, on=["id"])
            )
            df["pct_rank"] = df["rank"] / df.groupby("id")[
                "cell_id"
            ].transform("count")

            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=config.valid_ratio,
                random_state=config.seed,
            )
            idx_train, idx_valid = next(
                splitter.split(df, groups=df["ancestor_id"])
            )

            df_train = df.loc[idx_train].reset_index(drop=True).dropna()
            df_valid = df.loc[idx_valid].reset_index(drop=True).dropna()

            df_train_py = (
                df_train[df_train["cell_type"] == "code"]
                .drop("parent_id", axis=1)
                .dropna()
                .reset_index(drop=True)
            )
            df_valid_py = (
                df_valid[df_valid["cell_type"] == "code"]
                .drop("parent_id", axis=1)
                .dropna()
                .reset_index(drop=True)
            )
            df_train_md = (
                df_train[df_train["cell_type"] == "markdown"]
                .drop("parent_id", axis=1)
                .dropna()
                .reset_index(drop=True)
            )
            df_valid_md = (
                df_valid[df_valid["cell_type"] == "markdown"]
                .drop("parent_id", axis=1)
                .dropna()
                .reset_index(drop=True)
            )

            fts_train = get_features(df_train)
            fts_valid = get_features(df_valid)

            df_train.to_csv(config.input_dir / "train.csv", index=False)
            df_valid.to_csv(config.input_dir / "valid.csv", index=False)
            df_train_md.to_csv(config.input_dir / "train_md.csv", index=False)
            df_valid_md.to_csv(config.input_dir / "valid_md.csv", index=False)
            df_train_py.to_csv(config.input_dir / "train_py.csv", index=False)
            df_valid_py.to_csv(config.input_dir / "valid_py.csv", index=False)
            json.dump(
                fts_train, open(config.input_dir / "train_fts.json", "w")
            )
            json.dump(
                fts_valid, open(config.input_dir / "valid_fts.json", "w")
            )

            return (
                df_train,
                df_valid,
                df_train_md,
                df_valid_md,
                df_train_py,
                df_valid_py,
                fts_train,
                fts_valid,
                df_orders,
            )

        elif config.mode == "test":
            df_test = df.reset_index(drop=True).dropna()

            df_test["rank"] = df_test.groupby(["id", "cell_type"]).cumcount()
            df_test["pred"] = df_test.groupby(["id", "cell_type"])[
                "rank"
            ].rank(pct=True)
            df_test["pct_rank"] = 0

            df_test_py = (
                df_test[df_test["cell_type"] == "code"]
                .dropna()
                .reset_index(drop=True)
            )
            df_test_md = (
                df_test[df_test["cell_type"] == "markdown"]
                .dropna()
                .reset_index(drop=True)
            )

            fts_test = get_features(df_test)

            df_test.to_csv(config.input_dir / "test.csv", index=False)
            df_test_md.to_csv(config.input_dir / "test_md.csv", index=False)
            df_test_py.to_csv(config.input_dir / "test_py.csv", index=False)
            json.dump(fts_test, open(config.input_dir / "test_fts.json", "w"))

            return df_test, df_test_md, df_test_py, fts_test

    else:
        if config.mode == "train":
            df_train = pd.read_csv(config.input_dir / "train.csv").reset_index(
                drop=True
            )
            df_valid = pd.read_csv(config.input_dir / "valid.csv").reset_index(
                drop=True
            )
            df_train_md = pd.read_csv(
                config.input_dir / "train_md.csv"
            ).reset_index(drop=True)
            df_valid_md = pd.read_csv(
                config.input_dir / "valid_md.csv"
            ).reset_index(drop=True)
            df_train_py = pd.read_csv(
                config.input_dir / "train_py.csv"
            ).reset_index(drop=True)
            df_valid_py = pd.read_csv(
                config.input_dir / "valid_py.csv"
            ).reset_index(drop=True)
            fts_train = json.load(
                open(config.input_dir / "train_fts.json", "r")
            )
            fts_valid = json.load(
                open(config.input_dir / "valid_fts.json", "r")
            )
            df_orders = pd.read_csv(
                config.input_dir / "train_orders.csv",
                index_col="id",
                squeeze=True,
            ).str.split()
            return (
                df_train,
                df_valid,
                df_train_md,
                df_valid_md,
                df_train_py,
                df_valid_py,
                fts_train,
                fts_valid,
                df_orders,
            )

        elif config.mode == "test":
            df_test = pd.read_csv(config.input_dir / "test.csv").reset_index(
                drop=True
            )
            df_test_md = pd.read_csv(
                config.input_dir / "test_md.csv"
            ).reset_index(drop=True)
            df_test_py = pd.read_csv(
                config.input_dir / "test_py.csv"
            ).reset_index(drop=True)
            fts_test = json.load(open(config.input_dir / "test_fts.json", "r"))
            return df_test, df_test_md, df_test_py, fts_test


def read_notebook(path):
    return (
        pd.read_json(path, dtype={"cell_type": "category", "source": "str"})
        .assign(id=path.stem)
        .rename_axis("cell_id")
    )


def get_ranks(base, derived):
    return [base.index(d) for d in derived]


def clean_code(cell):
    return str(cell).replace("\\n", "\n")


def sample_cells(cells, sample_size=20):
    cells = [clean_code(cell) for cell in cells]
    if sample_size >= len(cells):
        return [cell[:200] for cell in cells]
    else:
        step = len(cells) / sample_size
        idx = 0
        samples = []
        while int(np.round(idx)) < len(cells):
            samples.append(cells[int(np.round(idx))])
            idx += step
        assert cells[0] in samples
        if cells[-1] not in samples:
            samples[-1] = cells[-1]
        return samples


def get_features(df):
    features = {}
    df = df.sort_values("rank").reset_index(drop=True)
    for idx, sub_df in tqdm(df.groupby("id")):
        total_md = sub_df[sub_df.cell_type == "markdown"].shape[0]
        code_sub_df = sub_df[sub_df.cell_type == "code"]
        total_code = code_sub_df.shape[0]
        codes = sample_cells(code_sub_df.source.values, 20)
        features[idx] = {
            "total_code": total_code,
            "total_md": total_md,
            "codes": codes,
        }
    return features
