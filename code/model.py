import torch
import torch.nn as nn
from torch.nn import DataParallel

import transformers as tx


class CodeRearranger(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.model = pretrained_model
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, x = self.model(
            ids, mask, token_type_ids=token_type_ids, return_dict=False
        )
        x = self.dropout(x)
        x = self.fc(x)
        y = torch.sigmoid(x)
        return y


def get_model(config):
    model_path = (
        "Salesforce/codet5-base"
        if config.basemodel == "codet5"
        else "microsoft/codebert-base"
    )

    tokenizer = tx.AutoTokenizer.from_pretrained(
        model_path,
        do_lower_case=False,  # "uncased" in config.prev_model
        is_split_into_words=True,
    )
    model = CodeRearranger(tx.AutoModel.from_pretrained(model_path))

    try:
        model.load_state_dict(
            torch.load(config.working_dir / config.basemodel / config.prev_model)
        )
    except:
        print(
            f"There is no {config.prev_model}, use base {config.basemodel} instead"
        )

    model = DataParallel(model, device_ids=[0, 1, 2, 3]).to(config.device)
    return tokenizer, model
