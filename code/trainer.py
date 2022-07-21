import gc
import os
from os import path as osp

import torch
from sklearn.metrics import mean_squared_error
from torch.cuda.amp import autocast
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        config,
        dataloaders,
        model,
        optimizer,
        criterion,
        scheduler,
        scaler,
        logger,
    ):
        self.config = config
        self.trainloader, self.validloader = dataloaders
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.scaler = scaler
        self.logger = logger

    def train_one_epoch(self):
        """
        Trains the model for 1 epoch
        """
        self.model.train()
        train_pbar = tqdm(
            enumerate(self.trainloader), total=len(self.trainloader)
        )
        train_preds, train_targets = [], []

        for bnum, data in train_pbar:
            ids = data[0].to(self.config.device)
            mask = data[1].to(self.config.device)
            fts = data[2].to(self.config.device)
            targets = data[-1].to(self.config.device)

            with autocast(enabled=True):
                outputs = self.model(ids=ids, mask=mask, fts=fts)

                loss = self.criterion(outputs, targets)

                loss_item = loss.item()
                self.wandb_log(train_batch_loss=loss_item)
                train_pbar.set_description(f"train loss: {loss_item:.4f}")

            self.scaler.scale(loss).backward()
            if bnum % self.config.accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            train_targets.extend(targets.cpu().detach().numpy().tolist())
            train_preds.extend(outputs.cpu().detach().numpy().tolist())

        del outputs, targets, ids, mask, fts, loss_item, loss
        gc.collect()
        torch.cuda.empty_cache()

        return train_preds, train_targets

    @torch.no_grad()
    def valid_one_epoch(self):
        """
        Validates the model for 1 epoch
        """
        self.model.eval()
        valid_pbar = tqdm(
            enumerate(self.validloader), total=len(self.validloader)
        )
        valid_preds, valid_targets = [], []

        for _, data in valid_pbar:
            ids = data[0].to(self.config.device)
            mask = data[1].to(self.config.device)
            fts = data[2].to(self.config.device)
            targets = data[-1].to(self.config.device)

            outputs = self.model(ids=ids, mask=mask, fts=fts).view(-1)

            valid_loss = self.criterion(outputs, targets)
            self.wandb_log(valid_batch_loss=valid_loss.item())
            valid_pbar.set_description(f"valid loss: {valid_loss.item():.4f}")

            valid_targets.extend(targets.cpu().detach().numpy().tolist())
            valid_preds.extend(outputs.cpu().detach().numpy().tolist())

        del outputs, targets, ids, mask, fts, valid_loss
        gc.collect()
        torch.cuda.empty_cache()

        return valid_preds, valid_targets

    def train(self, epochs: int = 10):
        self.logger.watch(self.model)

        """
        Low-effort alternative for doing the complete training and validation process
        """
        best_loss = int(1e9)
        for epoch in range(epochs):
            print(f"{'='*20} Epoch: {epoch+1} / {epochs} {'='*20}")

            train_preds, train_targets = self.train_one_epoch()
            train_mse = mean_squared_error(train_targets, train_preds)
            print(f"Training loss: {train_mse:.4f}")

            valid_preds, valid_targets = self.valid_one_epoch()
            valid_mse = mean_squared_error(valid_targets, valid_preds)
            print(f"Validation loss: {valid_mse:.4f}")

            self.wandb_log(train_mse=train_mse, valid_mse=valid_mse)

            if valid_mse < best_loss:
                best_loss = valid_mse
                save_path = (
                    self.config.working_dir
                    / self.config.base_model
                    / self.config.trial_name
                )
                self.save_model(save_path, f"ckpt_{epoch+1:03d}.pth")
                print(f"Saved model with val_loss: {best_loss:.4f}")

    def save_model(self, path, name, verbose=False):
        """
        Saves the model at the provided destination
        """
        try:
            if not osp.exists(path):
                os.makedirs(path)
        except:
            print("Errors encountered while making the output directory")

        torch.save(self.model.module.state_dict(), osp.join(path, name))
        if verbose:
            print(f"Model Saved at: {osp.join(path, name)}")

    def wandb_log(self, **kwargs):
        """
        Logs a key-value pair to W&B
        """
        for key, value in kwargs.items():
            self.logger.log({key: value})
