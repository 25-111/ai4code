from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from utils import read_notebook, get_ranks
from config import TrainConfig

NUM_TRAIN = 10000  #! remove for real train

paths_train = list((TrainConfig.data_dir / "train").glob("*.json"))[:NUM_TRAIN]
notebooks_train = [read_notebook(path) for path in tqdm(paths_train, desc="Train NBs")]
df = (
    pd.concat(notebooks_train)
    .set_index("id", append=True)
    .swaplevel()
    .sort_index(level="id", sort_remaining=False)
)

df_orders = pd.read_csv(
    TrainConfig.data_dir / "train_orders.csv",
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

df_ancestors = pd.read_csv(TrainConfig.data_dir / "train_ancestors.csv", index_col="id")

df = (
    df.reset_index()
    .merge(df_ranks, on=["id", "cell_id"])
    .merge(df_ancestors, on=["id"])
)

df["pct_rank"] = df["rank"] / df.groupby("id")["cell_id"].transform("count")


# split
splitter = GroupShuffleSplit(
    n_splits=1, test_size=TrainConfig.valid_ratio, random_state=0
)

idx_train, idx_valid = next(splitter.split(df, groups=df["ancestor_id"]))

df_train = df.loc[idx_train].reset_index(drop=True)
df_valid = df.loc[idx_valid].reset_index(drop=True)

df_train_md = df_train[df_train["cell_type"] == "markdown"].reset_index(drop=True)
df_valid_md = df_valid[df_valid["cell_type"] == "markdown"].reset_index(drop=True)
