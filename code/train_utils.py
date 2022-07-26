from torch.cuda.amp import GradScaler
from torch.nn import L1Loss, MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def yield_optimizer(model, config):
    if config.optim.lower() == "adamw":
        return AdamW(model.parameters(), lr=config.lr)


def yield_criterions():
    return MSELoss(), L1Loss()


def yield_scheduler(optimizer, config):
    return CosineAnnealingLR(
        optimizer=optimizer, T_max=50, eta_min=config.lr / 10
    )


def yield_scaler():
    return GradScaler()
