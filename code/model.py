# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2022-06-27 03:31:28
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2022-07-16 04:28:28

import torch
import torch.nn as nn
from torch.nn import DataParallel

import transformers as tx


def get_model(config):
    if config.base_model == "codebert":
        tokenizer = tx.AutoTokenizer.from_pretrained(
            "microsoft/codebert-base",
            do_lower_case=False,
            is_split_into_words=True,
        )
        model = RobertBased(
            tx.AutoModel.from_pretrained("microsoft/codebert-base")
        )
    elif config.base_model == "codet5":
        tokenizer = tx.AutoTokenizer.from_pretrained(
            "Salesforce/codet5-base",
            do_lower_case=False,
            is_split_into_words=True,
        )
        model = T5Based(tx.AutoModel.from_pretrained("Salesforce/codet5-base"))

    try:
        model.load_state_dict(torch.load(
            config.working_dir / config.base_model / config.prev_model
        ))
    except:
        print(f"There is no {config.prev_model}, use base {config.base_model} instead")

    if config.mode == "train":
        model = DataParallel(
            model, device_ids=[i for i in torch.cuda.device_count()]
        )
    return tokenizer, model.to(config.device)


class RobertBased(nn.Module):
    def __init__(self, pretrained_model, dropout_rate=0.5):
        super().__init__()
        self.model = pretrained_model
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, x = self.model(
            ids, mask, token_type_ids=token_type_ids, return_dict=False
        )
        x = self.dropout(x)
        x = self.fc(x)
        y = torch.sigmoid(x)
        return y


class T5Based(nn.Module):
    def __init__(self, pretrained_model, dropout_rate=0.5):
        super().__init__()
        self.model = pretrained_model
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(768, 1)

    def forward(self, ids, mask):
        _, x = self.model(ids, mask, return_dict=False)
        x = self.dropout(x)
        x = self.fc(x)
        y = torch.sigmoid(x)
        return y
