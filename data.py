import torch
from torch.utils.data import DataLoader, Dataset


class MarkdownDataset(Dataset):
    def __init__(self, df, max_len):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_len = max_len

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        ids = torch.LongTensor(inputs["input_ids"])
        mask = torch.LongTensor(inputs["attention_mask"])

        return ids, mask, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.df.shape[0]


def get_loaders(config):
    trainset = MarkdownDataset(df_train_mark, max_len=config.max_len)
    validset = MarkdownDataset(df_valid_mark, max_len=config.max_len)

    trainloader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    validloader = DataLoader(
        validset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return trainloader, validloader
