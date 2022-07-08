from datetime import datetime
from pytz import timezone
from pathlib import Path
import torch

# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# wandb_key = user_secrets.get_secret("WANDB_API_KEY")
# WANDB_KEY = "5f3589d7b951d748cc5a309b0b8c08aa7945ce52"


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
    max_len = 120
    num_epochs = 32
    num_workers = 8
    batch_size = 128
    lr = 3e-4
    seed = 42

    # Defaults
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now(timezone("Asia/Seoul")).strftime("%y%m%d-%H%M")
    trial_name = f"{timestamp}-{model_name}-{optim}-{loss}"
    data_dir = Path("../input/AI4Code/")
    log_dir = Path("./working/")


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
