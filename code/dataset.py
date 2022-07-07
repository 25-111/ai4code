import torch
import torch.utils.data as dt


def read_data(data, config):
    return (d.to(config.device) for d in data[:-1]), data[-1].to(config.device)


class NotebookDataset(dt.Dataset):
    def __init__(self, df, max_len, tokenizer, config):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        # self.fts = fts
        self.tokenizer = tokenizer
        self.config = config

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]

        inputs = self.tokenizer.encode_plus(
            sample.source,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        # code_inputs = self.tokenizer.batch_encode_plus(
        #     [str(x) for x in self.fts[sample.id]["codes"]],
        #     add_special_tokens=True,
        #     max_length=23,
        #     padding="max_length",
        #     truncation=True,
        # )

        ids = torch.LongTensor(inputs["input_ids"])
        mask = torch.LongTensor(inputs["attention_mask"])
        token_type_ids = torch.LongTensor(inputs["token_type_ids"])

        if self.config.mode == "train":
            target = torch.FloatTensor([sample.pct_rank])
            return ids, mask, token_type_ids, target
        else:
            return ids, mask, token_type_ids

    def __len__(self):
        return self.df.shape[0]
