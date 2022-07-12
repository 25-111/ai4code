from datetime import datetime
from pathlib import Path

from pytz import timezone


class Config:
    # Defaults
    device = "cuda"
    input_dir = Path("../input/AI4Code/")
    working_dir = Path("../working/")

    # Model
    prev_model = None  # Path("0712-1900-from(codebert-base).pth")

    # Train
    optim = ["AdamW"][0]
    loss = ["MSE"][0]
    valid_ratio = 0.1
    max_len = 256
    num_epochs = 10
    num_workers = 8
    batch_size = 192
    lr = 3e-4
    seed = 42

    # Log
    timestamp = datetime.now(timezone("Asia/Seoul")).strftime("%m%d-%H%M")
    trial_name = (
        f"{timestamp}-from({str(prev_model)[:9]})"
        if prev_model is not None
        else f"{timestamp}-from(codebert-base)"
    )


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
