# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2022-07-06 04:27:28
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2022-07-11 03:58:16

import os
from os import path as osp
import gc
from tqdm import tqdm
import wandb
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import mean_squared_error
from config import Config, WandbConfig


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
    ):
        self.train_loader, self.valid_loader = dataloaders
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.model = model
        self.scaler = scaler
        self.device = config.device

    def train_one_epoch(self):
        """
        Trains the model for 1 epoch
        """
        self.model.train()
        train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        train_preds, train_targets = [], []

        for bnum, data in train_pbar:
            ids = data[0].to(self.device)
            mask = data[1].to(self.device)
            ttis = data[2].to(self.device)
            targets = data[-1].to(self.device)

            with autocast(enabled=True):
                outputs = self.model(ids=ids, mask=mask, token_type_ids=ttis)

                loss = self.criterion(outputs, targets)
                loss_item = loss.item()

                self.wandb_log(train_batch_loss=loss_item)

                train_pbar.set_description(f"train loss: {loss_item:.4f}")

                # self.scaler.scale(loss).backward()
                # self.scaler.step(self.optimizer)
                # self.scaler.update()
                loss.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()
                self.scheduler.step()

            train_targets.extend(targets.cpu().detach().numpy().tolist())
            train_preds.extend(outputs.cpu().detach().numpy().tolist())

        # Tidy
        del outputs, targets, ids, mask, ttis, loss_item, loss
        gc.collect()
        torch.cuda.empty_cache()

        return train_preds, train_targets

    @torch.no_grad()
    def valid_one_epoch(self):
        """
        Validates the model for 1 epoch
        """
        self.model.eval()
        valid_pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))
        valid_preds, valid_targets = [], []

        for bnum, data in valid_pbar:
            ids = data[0].to(self.device)
            mask = data[1].to(self.device)
            ttis = data[2].to(self.device)
            targets = data[-1].to(self.device)

            outputs = self.model(ids=ids, mask=mask, token_type_ids=ttis).view(-1)
            valid_loss = self.criterion(outputs, targets)

            self.wandb_log(valid_batch_loss=valid_loss.item())

            valid_pbar.set_description(f"val_loss: {valid_loss.item():.4f}")

            valid_targets.extend(targets.cpu().detach().numpy().tolist())
            valid_preds.extend(outputs.cpu().detach().numpy().tolist())

        # Tidy
        del outputs, targets, ids, mask, ttis, valid_loss
        gc.collect()
        torch.cuda.empty_cache()

        return valid_preds, valid_targets

    def train(
        self,
        epochs: int=10,
        output_dir: str="./working/",
    ):
        config, wandb_config = Config(), WandbConfig()
        wandb.init(
            project="ai4code",
            entity="25111",
            name=config.trial_name,
            config=wandb_config,
            dir=config.log_dir,
        )
        wandb.watch(self.model)

        """
        Low-effort alternative for doing the complete training and validation process
        """
        best_loss = int(1e9)
        best_preds = None
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
                save_path = osp.join(output_dir, config.trial_name)
                self.save_model(save_path, f"model_{epoch}.pth")
                print(f"Saved model with val_loss: {best_loss:.4f}")
                wandb.save(osp.join(save_path, f"model_{epoch}.pth"))

        wandb.run.finish()

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
            wandb.log({key: value})
