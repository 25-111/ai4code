import wandb
import torch
from train import train
from config import BertLargeConfig, DistillBertConfig


def main(arg):
    MODEL_CONFIG = BertLargeConfig()

    WANDB_CONFIG = {
        "bs_train": MODEL_CONFIG.bs_train,
        "bs_valid": MODEL_CONFIG.bs_valid,
        "num_epochs": MODEL_CONFIG.num_epochs,
        "model": MODEL_CONFIG.model_name,
        "max_len": MODEL_CONFIG.max_len,
        "lr": MODEL_CONFIG.lr,
        "num_workers": 8,
        "optim": "AdamW",
        "loss": "MSELoss",
        "device": "cuda",
        "T_0": 20,
        "min_eta": 1e-4,
        "infra": "Kaggle",
        "competition": "ai4code",
        "_wandb_kernel": "tanaym",
    }

    if MODEL_CONFIG.wandb:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        wb_key = user_secrets.get_secret("WANDB_API_KEY")

        wandb.login(key=wb_key)

        run = wandb.init(
            project="pytorch",
            config=WANDB_CONFIG,
            group="nlp",
            job_type="train",
        )

    criterion = torch.nn.MSELoss()

    train()
