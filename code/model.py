import torch
import torch.nn as nn
from torch.nn import DataParallel

import transformers as tx


def get_model(config):
    if config.base_model == "codebert":
        model_path = "microsoft/codebert-base"
    elif config.base_model == "graphcodebert":
        model_path = "microsoft/graphcodebert-base"
    elif config.base_model == "codet5":
        model_path = "Salesforce/codet5-base"

    tokenizer = tx.AutoTokenizer.from_pretrained(
        model_path,
        do_lower_case=False,
        is_split_into_words=True,
    )
    model = NotebookArranger(tx.AutoModel.from_pretrained(model_path))

    try:
        model.load_state_dict(
            torch.load(
                config.working_dir / config.base_model / config.prev_model
            )
        )
    except:
        print(
            f"There is no {config.prev_model}, use base {config.base_model} instead"
        )

    if config.mode == "train":
        model = DataParallel(model, device_ids=[0, 1, 2, 3])
    return tokenizer, model.to(config.device)


class NotebookArranger(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.model = pretrained_model
        self.fc = nn.Linear(769, 1)

    def forward(self, ids, mask, fts):
        x = self.model(ids, mask)[0]
        y = self.fc(torch.cat((x[:, 0, :], fts), 1))
        return y
