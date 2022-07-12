from datetime import datetime
from pathlib import Path

from pytz import timezone


class Config:
    # Model
    available_models = [
        "bert-large-uncased",
        "bert-large-cased",
        "distilbert-base-uncased",
        "microsoft/codebert-base",
        "roberta-base",
        "roberta-large",
        "albert-base-v2",
    ]
    model_name = available_models[3]

    # Train
    optim = ["AdamW"][0]
    loss = ["MSE"][0]
    valid_ratio = 0.1
    max_len = 256
    num_epochs = 2
    num_workers = 8
    batch_size = 192
    lr = 3e-4
    seed = 42

    # Defaults
    device = "cuda"
    timestamp = datetime.now(timezone("Asia/Seoul")).strftime("%y%m%d-%H%M%S")
    trial_name = f"{timestamp}-{model_name.replace('/', '_')}-{optim}-{loss}"
    input_dir = Path("../input/AI4Code/")
    working_dir = Path("../working/")


class WandbConfig:
    model = Config.model_name
    optim = Config.optim
    loss = Config.loss
    max_len = Config.max_len
    num_epochs = Config.num_epochs
    num_workers = Config.num_workers
    batch_size = Config.batch_size
    lr = Config.lr
    device = Config.device
