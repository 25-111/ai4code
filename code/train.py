import sys, warnings
from tqdm import tqdm
import wandb
import numpy as np
import torch
import transformers as tx
from preprocess import preprocess, get_features

# TODO: read_data 처리 (아마 Dataset에서 처리하는 것이 나을 듯)
from dataset import NotebookDataset, read_data
from model import get_model
from config import Config, WandbConfig
from metric import calc_kendall_tau


def main():
    # Configuration
    config = Config()
    config.mode = "train"

    if config.wandb_key:
        wandb_config = WandbConfig()
        wandb.login(key=config.wandb_key)

        run = wandb.init(
            project="ai4code",
            entity="nciaproject",
            config=wandb_config,
            dir=config.log_dir,
        )

    # Loading Model
    tokenizer, model = get_model(config)

    # Loading Data
    use_pin_mem = config.device.startswith("cuda")

    df_train, df_train_md, df_valid, df_valid_md, df_orders = preprocess(config)
    fts_train, fts_valid = get_features(df_train), get_features(df_valid)

    trainset = NotebookDataset(
        df_train_md,
        max_len=config.max_len,
        max_len_md=config.max_len_md,
        fts=fts_train,
        tokenizer=tokenizer,
    )
    validset = NotebookDataset(
        df_valid_md,
        max_len=config.max_len,
        max_len_md=config.max_len_md,
        fts=fts_valid,
        tokenizer=tokenizer,
    )
    data_collator = tx.DataCollatorWithPadding(tokenizer=tokenizer)

    # Setting Train
    trainarg = tx.TrainingArguments(
        output_dir=config.result_dir,
        do_train=True,
        do_predict=True,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.num_epochs,
        logging_dir=config.log_dir,
        seed=config.seed,
        dataloader_num_workers=config.num_workers,
        load_best_model_at_end=True,
        optim=config.optim,
        report_to="wandb",
        dataloader_pin_memory=use_pin_mem,
    )
    trainer = tx.Trainer(
        model=model,
        args=trainarg,
        train_dataset=trainset,
        eval_dataset=validset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        # callbacks=[tx.EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    trainer.train()

    if config.wandb_key:
        run.finish()


def validate(model, validloader, config):
    model.eval()

    tbar = tqdm(validloader, file=sys.stdout)

    preds, labels = [], []
    with torch.no_grad():
        for _, data in enumerate(tbar):
            inputs, labels = read_data(data, config)

            pred = model(*inputs)

            labels.append(labels.detach().cpu().numpy().ravel())
            preds.append(pred.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)


if __name__ == "__main__":
    main()
