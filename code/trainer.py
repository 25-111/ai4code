# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2022-07-06 04:27:28
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2022-07-07 04:04:27

import wandb
from tqdm import tqdm
import torch
import gc
from sklearn.metrics import mean_squared_error
import os
from os import path as osp
from torch.cuda.amp import autocast
from config import WandbConfig

class Trainer:
    def __init__(self,
        config,
        dataloaders,
        optimizer,
        model,
        loss_fn,
        scheduler,
        scaler=None,
        device="cuda"
    ):
        self.train_loader, self.valid_loader = dataloaders
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.model = model
        self.device = 'cuda'
        self.scaler = scaler
        self.config = config


    def train_one_epoch(self):
        """
        Trains the model for 1 epoch
        """
        self.model.train()
        train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        train_preds, train_targets = [], []

        for bnum, cache in train_pbar:
            ids = self._convert_if_not_tensor(cache[0], dtype=torch.long)
            mask = self._convert_if_not_tensor(cache[1], dtype=torch.long)
            # ttis = self._convert_if_not_tensor(cache[2], dtype=torch.long)
            targets = self._convert_if_not_tensor(cache[2], dtype=torch.float)
            
            with autocast(enabled=True):
                outputs = self.model(ids=ids, mask=mask).view(-1)
                
                loss = self.loss_fn(outputs, targets)
                loss_itm = loss.item()
                
                self.wandb_log(train_batch_loss=loss_itm)
                
                train_pbar.set_description(f'train loss: {loss_itm:.4f}')

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            train_targets.extend(targets.cpu().detach().numpy().tolist())
            train_preds.extend(outputs.cpu().detach().numpy().tolist())
        
        # Tidy
        del outputs, targets, ids, mask, ttis, loss_itm, loss
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

        for idx, cache in valid_pbar:
            ids = self._convert_if_not_tensor(cache[0], dtype=torch.long)
            mask = self._convert_if_not_tensor(cache[1], dtype=torch.long)
            ttis = self._convert_if_not_tensor(cache[2], dtype=torch.long)
            targets = self._convert_if_not_tensor(cache[3], dtype=torch.float)

            outputs = self.model(
                ids=ids, mask=mask, token_type_ids=ttis
            ).view(-1)
            valid_loss = self.loss_fn(outputs, targets)
            
            self.wandb_log(valid_batch_loss=valid_loss.item())
            
            valid_pbar.set_description(f"val_loss: {valid_loss.item():.4f}")

            valid_targets.extend(targets.cpu().detach().numpy().tolist())
            valid_preds.extend(outputs.cpu().detach().numpy().tolist())

        # Tidy
        del outputs, targets, ids, mask, ttis, valid_loss
        gc.collect()
        torch.cuda.empty_cache()
        
        return valid_preds, valid_targets

    def train(self,
        epochs: int = 10,
        output_dir: str = "/kaggle/working/",
        custom_name: str = 'model.pth'
    ):
        self.run = wandb.init(
            project="ai4code",
            entity="25111",
            name="test",
            config=WandbConfig()
        )
        wandb.watch(self.model)
        """
        Low-effort alternative for doing the complete training and validation process
        """
        best_loss = int(1e+9)
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
                self.save_model(output_dir, f"model_{epoch}.pth")
                print(f"Saved model with val_loss: {best_loss:.4f}")

        run.finish()
    def save_model(self, path, name, verbose=False):
        """
        Saves the model at the provided destination
        """
        try:
            if not osp.exists(path):
                os.makedirs(path)
        except:
            print("Errors encountered while making the output directory")

        torch.save(self.model.state_dict(), osp.join(path, name))
        if verbose:
            print(f"Model Saved at: {osp.join(path, name)}")

    def _convert_if_not_tensor(self, x, dtype):
        if self._tensor_check(x):
            return x.to(self.device, dtype=dtype)
        else:
            return torch.tensor(x, dtype=dtype, device=self.device)

    def _tensor_check(self, x):
        return isinstance(x, torch.Tensor)

    def wandb_log(self, **kwargs):
        """
        Logs a key-value pair to W&B
        """
        for key, value in kwargs.items():
            wandb.log({key: value})
