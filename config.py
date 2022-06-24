from pathlib import Path
import torch

# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# wandb_key = user_secrets.get_secret("WANDB_API_KEY")
WANDB_KEY = None


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
    model_name = available_models[2]

    # Train
    optim = "AdamW"
    loss = "MSE"
    valid_ratio = 0.1
    max_len = 512
    max_len_md = 64
    num_epochs = 2
    num_workers = 8
    batch_size = 32
    lr = 3e-4
    seed = 1234

    # Defaults
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("/data/AI4Code")  # Path("../input/AI4Code")
    log_dir = Path("/data/AI4Code/log")
    result_dir = Path("/data/AI4Code/result")
    wandb_key = WANDB_KEY


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
