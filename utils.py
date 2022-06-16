from bisect import bisect
import pandas as pd
import wandb

# preprocessing
def read_notebook(path):
    return (
        pd.read_json(path, dtype={"cell_type": "category", "source": "str"})
        .assign(id=path.stem)
        .rename_axis("cell_id")
    )


def get_ranks(base, derived):
    return [base.index(d) for d in derived]


# optimizer
def adjust_lr(optimizer, epoch):
    if epoch < 1:
        lr = 5e-5
    elif epoch < 2:
        lr = 1e-3
    elif epoch < 5:
        lr = 1e-4
    else:
        lr = 1e-5

    for p in optimizer.param_groups:
        p["lr"] = lr
    return lr


# evaluation
def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions


def calc_kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max


# logging
def wandb_log(**kwargs):
    """Logs a key-value pair to W&B"""
    for k, v in kwargs.items():
        wandb.log({k: v})
