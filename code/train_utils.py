from torch.cuda.amp import GradScaler
from torch.nn import L1Loss, MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def yield_optimizer(model, config):
    if config.optim.lower() == "adamw":
        return AdamW(model.parameters(), lr=config.lr)


def yield_criterion(config):
    if config.loss.lower() == "mse":
        return MSELoss()
    elif config.loss.lower() == "l1":
        return L1Loss()


def yield_scheduler(optimizer):
    return CosineAnnealingLR(optimizer=optimizer, T_max=50, eta_min=3e-5)


def yield_scaler():
    return GradScaler()
