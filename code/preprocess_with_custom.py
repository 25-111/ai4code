import json

import pandas as pd
from preprocess import get_features
from sklearn.model_selection import GroupShuffleSplit


def preprocess_with_custom(config):
    if config.mode == "train":
        df_train = pd.read_csv(config.input_dir / "train.csv").reset_index(
            drop=True
        )
        df_valid = pd.read_csv(config.input_dir / "valid.csv").reset_index(
            drop=True
        )
        df = df_train.append(df_valid)

        df_orders = pd.read_csv(
            config.input_dir / "train_orders.csv",
            index_col="id",
            squeeze=True,
        ).str.split()

        df_custom = pd.read_csv(config.input_dir / "custom.csv").reset_index(
            drop=True
        )
        df_custom[["id", "cell_id", "ancestor_id", "parent_id"]] = df_custom[
            ["id", "cell_id", "ancestor_id", "parent_id"]
        ].astype(str)
        df_custom_orders = pd.read_csv(
            config.input_dir / "custom_orders.csv",
            index_col="id",
            squeeze=True,
        ).str.split()

        df = df.append(df_custom)
        df_orders = df_orders.append(df_custom_orders)

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

        df_train.to_csv(
            config.input_dir / "train_with_custom.csv", index=False
        )
        df_valid.to_csv(
            config.input_dir / "valid_with_custom.csv", index=False
        )
        df_train_md.to_csv(
            config.input_dir / "train_with_custom_md.csv", index=False
        )
        df_valid_md.to_csv(
            config.input_dir / "valid_with_custom_md.csv", index=False
        )
        json.dump(
            fts_train,
            open(config.input_dir / "train_with_custom_fts.json", "w"),
        )
        json.dump(
            fts_valid,
            open(config.input_dir / "valid_with_custom_fts.json", "w"),
        )

        return (
            df_train,
            df_valid,
            df_train_md,
            df_valid_md,
            fts_train,
            fts_valid,
            df_orders,
        )
