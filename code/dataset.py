import torch
from torch.utils.data import Dataset

import transformers as tx


class NotebookDataset(Dataset):
    def __init__(self, df, fts, config):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.fts = fts
        self.config = config

        self.tokenizer = tx.AutoTokenizer.from_pretrained(config.model_path)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        inputs = self.tokenizer.encode_plus(
            item.source,
            None,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            # max_length=self.config.md_max_len,
        )
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(x) for x in self.fts[item.id]["codes"]],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            # max_length=self.config.py_max_len,
        )

        n_md = self.fts[item.id]["total_md"]
        n_code = self.fts[item.id]["total_md"]

        if n_md + n_code == 0:
            fts = torch.FloatTensor([0])
        else:
            fts = torch.FloatTensor([n_md / (n_md + n_code)])

        ids = inputs["input_ids"]
        for x in code_inputs["input_ids"]:
            ids.extend(x[:-1])
        ids = ids[: self.config.total_max_len]
        if len(ids) != self.config.total_max_len:
            ids = ids + [
                self.tokenizer.pad_token_id,
            ] * (self.config.total_max_len - len(ids))
        ids = torch.LongTensor(ids)

        mask = inputs["attention_mask"]
        for x in code_inputs["attention_mask"]:
            mask.extend(x[:-1])
        mask = mask[: self.config.total_max_len]
        if len(mask) != self.config.total_max_len:
            mask = mask + [
                self.tokenizer.pad_token_id,
            ] * (self.config.total_max_len - len(mask))
        mask = torch.LongTensor(mask)

        if self.config.mode == "train":
            target = torch.FloatTensor([item.pct_rank])
            return ids, mask, fts, target
        else:
            return ids, mask, fts

    def __len__(self):
        return self.df.shape[0]
