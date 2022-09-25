import json
from os import path as osp

import pandas as pd
from preprocess import get_features
from sklearn.model_selection import GroupShuffleSplit


def preprocess_custom(config):
    if config.mode == "train":
        if not osp.exists(config.input_dir / f"{config.mode}_custom.csv"):
            print("Preprocessing..: Start")
            df = pd.read_csv(config.input_dir / "custom.csv").reset_index(
                drop=True
            )
            df = df.astype(
                {
                    "id": str,
                    "cell_id": str,
                    "ancestor_id": str,
                    "parent_id": str,
                }
            )

            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=config.valid_ratio,
                random_state=config.seed,
            )
            idx_train, idx_valid = next(
                splitter.split(df, groups=df["ancestor_id"])
            )

            df_train = df.iloc[idx_train].reset_index(drop=True).dropna()
            df_valid = df.iloc[idx_valid].reset_index(drop=True).dropna()

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

            df_train.to_csv(config.input_dir / "train_custom.csv", index=False)
            df_valid.to_csv(config.input_dir / "valid_custom.csv", index=False)
            df_train_md.to_csv(
                config.input_dir / "train_custom_md.csv", index=False
            )
            df_valid_md.to_csv(
                config.input_dir / "valid_custom_md.csv", index=False
            )
            json.dump(
                fts_train,
                open(config.input_dir / "train_custom_fts.json", "w"),
            )
            json.dump(
                fts_valid,
                open(config.input_dir / "valid_custom_fts.json", "w"),
            )
            print("Preprocessing..: Done!")

        else:
            df_train = pd.read_csv(
                config.input_dir / "train_custom.csv"
            ).reset_index(drop=True)
            df_valid = pd.read_csv(
                config.input_dir / "valid_custom.csv"
            ).reset_index(drop=True)
            df_train_md = pd.read_csv(
                config.input_dir / "train_custom_md.csv"
            ).reset_index(drop=True)
            df_valid_md = pd.read_csv(
                config.input_dir / "valid_custom_md.csv"
            ).reset_index(drop=True)
            fts_train = json.load(
                open(config.input_dir / "train_custom_fts.json", "r")
            )
            fts_valid = json.load(
                open(config.input_dir / "valid_custom_fts.json", "r")
            )

        df_orders = pd.read_csv(
            config.input_dir / "custom_orders.csv",
            index_col="id",
            squeeze=True,
        ).str.split()

        return (
            df_train,
            df_valid,
            df_train_md,
            df_valid_md,
            fts_train,
            fts_valid,
            df_orders,
        )
