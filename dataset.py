import torch
from torch.utils.data import DataLoader, Dataset
from preprocess import get_features


class NotebookDataset(Dataset):
    def __init__(self, df, max_len, max_len_md, fts, tokenizer):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.max_len_md = max_len_md
        self.fts = fts
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.max_len_md,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(x) for x in self.fts[row.id]["codes"]],
            add_special_tokens=True,
            max_length=23,
            padding="max_length",
            truncation=True,
        )
        n_md = self.fts[row.id]["total_md"]
        n_code = self.fts[row.id]["total_md"]
        if n_md + n_code == 0:
            fts = torch.FloatTensor([0])
        else:
            fts = torch.FloatTensor([n_md / (n_md + n_code)])

        ids = inputs["input_ids"]
        for x in code_inputs["input_ids"]:
            ids.extend(x[:-1])
        ids = ids[: self.max_len]
        if len(ids) != self.max_len:
            ids = ids + [
                self.tokenizer.pad_token_id,
            ] * (self.max_len - len(ids))
        ids = torch.LongTensor(ids)
        assert len(ids) == self.max_len

        mask = inputs["attention_mask"]
        for x in code_inputs["attention_mask"]:
            mask.extend(x[:-1])
        mask = mask[: self.max_len]
        if len(mask) != self.max_len:
            mask = mask + [
                self.tokenizer.pad_token_id,
            ] * (self.max_len - len(mask))
        mask = torch.LongTensor(mask)

        return ids, mask, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.df.shape[0]


def get_loaders(df_train, df_train_md, df_valid, df_valid_md, tokenizer, config):
    use_pin_mem = config.device.startswith("cuda")

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

    trainloader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_pin_mem,
        drop_last=True,
    )
    validloader = DataLoader(
        validset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_mem,
        drop_last=False,
    )
    return trainloader, validloader


def read_data(data, config):
    return (d.to(config.device) for d in data[:-1]), data[-1].to(config.device)
