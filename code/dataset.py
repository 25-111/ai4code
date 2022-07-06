import torch
import torch.utils.data as dt


def read_data(data, config):
    return (d.to(config.device) for d in data[:-1]), data[-1].to(config.device)


class NotebookDataset(dt.Dataset):
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

        # ttids = inputs["token_type_ids"]
        # for x in code_inputs["token_type_ids"]:
        #     ttids.extend(x[:-1])
        # ttids = ttids[: self.max_len]
        # if len(ttids) != self.max_len:
        #     ttids = ttids + [
        #         self.tokenizer.pad_token_id,
        #     ] * (self.max_len - len(ttids))
        # ttids = torch.LongTensor(ttids)
        # data = (ids, mask, torch.FloatTensor([row.pct_rank]))
        # return read_data(data, config)
        return ids, mask, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.df.shape[0]
