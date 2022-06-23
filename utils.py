import wandb


def adjust_lr(optimizer, epoch):
    if epoch < 1:
        lr = 5e-5
    elif epoch < 2:
        lr = 1e-3
    elif epoch < 5:
        lr = 1e-4
    else:
        lr = 1e-5

    for p in optimizer.param_groups:
        p["lr"] = lr
    return lr


def wandb_log(**kwargs):
    try:
        for k, v in kwargs.items():
            wandb.log({k: v})
    except:
        pass
