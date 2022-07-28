from datetime import datetime
from pathlib import Path

import torch
from pytz import timezone


class Config:
    # Data
    custom_data = True

    # Model
    prev_model = Path("graphcodebert-base/graphcodebert-base.pth")
    adjustment = "scaler-fts-lr*.5-custom_data"

    # Train
    optim = ["AdamW"][0]
    l2_weight, l1_weight = 1, 0
    valid_ratio = 0.1
    md_max_len = 128
    py_max_len = 23
    total_max_len = 512
    num_epochs = 3
    num_workers = 8
    batch_size = 64
    lr = 1.5e-4
    accum_steps = 4
    seed = 42

    # Log
    timestamp = datetime.now(timezone("Asia/Seoul")).strftime("%m%d-%H%M")
    try:
        base_model = str(prev_model).split("/")[0].split("-")[3]
    except:
        base_model = str(prev_model).split("/")[0].split("-")[0]
    trial_name = (
        f"{timestamp}-{base_model}-fts-from-base"
        if str(prev_model).endswith("base.pth")
        else f"{timestamp}-{base_model}-fts-from-{str(prev_model)[:9]}"
    )
    trial_name += f"-{adjustment}" if adjustment else ""

    # Defaults
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dir = Path("../input/AI4Code/")
    working_dir = Path("../working/")
    if base_model == "codebert":
        model_path = "microsoft/codebert-base"
    elif base_model == "graphcodebert":
        model_path = "microsoft/graphcodebert-base"
    elif base_model == "codet5":
        model_path = "Salesforce/codet5-base"


class WandbConfig:
    model = Config.prev_model
    optim = Config.optim
    l1_weight = Config.l1_weight
    l2_weight = Config.l2_weight
    md_max_len = Config.md_max_len
    py_max_len = Config.py_max_len
    total_max_len = Config.total_max_len
    num_epochs = Config.num_epochs
    num_workers = Config.num_workers
    batch_size = Config.batch_size
    lr = Config.lr
    accum_steps = Config.accum_steps
