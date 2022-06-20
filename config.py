from pathlib import Path
import torch

# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# wandb_key = user_secrets.get_secret("WANDB_API_KEY")
wandb_key = None


class Config:
    model_name = "bert-large"  # ["bert-large", "distill-beart"]
    optim = "AdamW"
    loss = "MSE"
    valid_ratio = 0.1
    max_len = 120
    num_epochs = 2
    num_workers = 8
    batch_size = 32
    lr = 3e-4

    # Defaults
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path("/data/AI4Code")  # Path("../input/AI4Code")
    wandb_key = wandb_key


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
