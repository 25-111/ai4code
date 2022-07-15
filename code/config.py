from datetime import datetime
from pathlib import Path

from pytz import timezone


class Config:
    # Defaults
    device = "cuda"
    input_dir = Path("../input/AI4Code/")
    working_dir = Path("../working/")

    # Data
    data_type = ["all", "md", "py"][2]

    # Model
    prev_model = Path("codebert-base/codebert-base.pth")
    adjustment = "scaler"

    # Train
    optim = ["AdamW"][0]
    loss = ["MSE"][0]
    valid_ratio = 0.1
    max_len = 256
    num_epochs = 3
    num_workers = 8
    batch_size = 192
    lr = 3e-4
    accum_steps = 4
    seed = 42

    # Log
    timestamp = datetime.now(timezone("Asia/Seoul")).strftime("%m%d-%H%M")
    base_model = "codebert" if "codebert" in str(prev_model) else "codet5"
    trial_name = (
        f"{timestamp}-{data_type}-{base_model}-from-{str(prev_model)[:9]}"
        if not str(prev_model).endswith("base.pth")
        else f"{timestamp}-{data_type}-{base_model}-from-{base_model}-base"
    )
    trial_name += f"-{adjustment}" if adjustment else ""


class WandbConfig:
    model = Config.prev_model
    optim = Config.optim
    loss = Config.loss
    max_len = Config.max_len
    num_epochs = Config.num_epochs
    num_workers = Config.num_workers
    batch_size = Config.batch_size
    lr = Config.lr
    device = Config.device
