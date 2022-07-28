import gc
import os
import sys
from os import path as osp

import torch
from metric import calc_kendall_tau
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
        criterions,
        scheduler,
        scaler,
        df_valid,
        df_orders,
        logger,
    ):
        self.config = config
        self.trainloader, self.validloader = dataloaders
        self.model = model
        self.optimizer = optimizer
        self.l2_loss, self.l1_loss = criterions
        self.scheduler = scheduler
        self.scaler = scaler
        self.df_valid = df_valid
        self.df_orders = df_orders
        self.logger = logger

    def train(self, epochs):
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

            self.df_valid["pred"] = self.df_valid.groupby(["id", "cell_type"])[
                "rank"
            ].rank(pct=True)
            self.df_valid.loc[
                self.df_valid["cell_type"] == "markdown", "pred"
            ] = valid_preds
            pred_orders = (
                self.df_valid.sort_values("pred")
                .groupby("id")["cell_id"]
                .apply(list)
            )
            kendall_tau = calc_kendall_tau(
                self.df_orders.loc[pred_orders.index], pred_orders
            )
            print(f"Prediction Kendall Tau: {kendall_tau:.4f}")

            self.wandb_log(
                train_mse=train_mse,
                valid_mse=valid_mse,
                kendall_tau=kendall_tau,
            )

            if valid_mse < best_loss:
                best_loss = valid_mse
                save_path = (
                    self.config.working_dir
                    / self.config.base_model
                    / self.config.trial_name
                )
                self.save_model(save_path, f"ckpt_{epoch+1:03d}.pth")
                print(f"Saved model with val_loss: {best_loss:.4f}")

    def train_one_epoch(self):
        """
        Trains the model for 1 epoch
        """
        self.model.train()
        train_pbar = tqdm(
            self.trainloader, total=len(self.trainloader), file=sys.stdout
        )
        train_preds, train_targets = [], []

        for bnum, data in enumerate(train_pbar):
            ids = data[0].to(self.config.device)
            mask = data[1].to(self.config.device)
            fts = data[2].to(self.config.device)
            targets = data[-1].to(self.config.device)

            with autocast(enabled=True):
                preds = self.model(ids=ids, mask=mask, fts=fts)

                loss = self.config.l2_weight * self.l2_loss(
                    preds, targets
                ) + self.config.l1_weight * self.l1_loss(preds, targets)

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
            train_preds.extend(preds.cpu().detach().numpy().tolist())

        del preds, targets, ids, mask, fts, loss_item, loss
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
            self.validloader, total=len(self.validloader), file=sys.stdout
        )
        valid_preds, valid_targets = [], []

        for data in valid_pbar:
            ids = data[0].to(self.config.device)
            mask = data[1].to(self.config.device)
            fts = data[2].to(self.config.device)
            targets = data[-1].to(self.config.device)

            preds = self.model(ids=ids, mask=mask, fts=fts).view(-1)

            loss = self.config.l2_weight * self.l2_loss(
                preds, targets
            ) + self.config.l1_weight * self.l1_loss(preds, targets)

            loss_item = loss.item()
            self.wandb_log(valid_batch_loss=loss_item)
            valid_pbar.set_description(f"valid loss: {loss_item:.4f}")

            valid_targets.extend(targets.cpu().detach().numpy().tolist())
            valid_preds.extend(preds.cpu().detach().numpy().tolist())

        del preds, targets, ids, mask, fts, loss_item, loss
        gc.collect()
        torch.cuda.empty_cache()

        return valid_preds, valid_targets

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
