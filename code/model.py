# -*- coding: utf-8 -*-
# @Author: Yedarm Seong
# @Date:   2022-06-27 03:31:28
# @Last Modified by:   Yedarm Seong
# @Last Modified time: 2022-07-11 02:06:33

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
            ids,
            mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        x = self.dropout(x)
        x = self.fc(x)
        y = torch.sigmoid(x)
        return y


def get_model(config):
    # TODO: rename model_name to model_path for training on msj server
    # if config.model_name.startswith("bert"):
    #     tokenizer = tx.AutoTokenizer.from_pretrained(
    #         config.model_name, do_lower_case=config.model_name.endswith("uncased")
    #     )
    #     # tokenizer = tx.BertTokenizer.from_pretrained(
    #     #     config.model_name, do_lower_case=config.model_name.endswith("uncased")
    #     # )
    #     model = tx.AutoModel.from_pretrained(config.model_name)
    #     # model = tx.BertModel.from_pretrained(config.model_name)
    # elif config.model_name.startswith("distill-bert"):
    #     tokenizer = tx.DistilBertTokenizer.from_pretrained(
    #         config.model_name, do_lower_case=config.model_name.endswith("uncased")
    #     )
    #     model = tx.DistilBertModel.from_pretrained(config.model_name)
    # elif config.model_name.startswith("roberta"):
    #     tokenizer = tx.RobertaTokenizer.from_pretrained(
    #         config.model_name, do_lower_case=config.model_name.endswith("uncased")
    #     )
    #     model = tx.RobertaModel.from_pretrained(config.model_name)
    # elif config.model_name.startswith("albert"):
    #     tokenizer = tx.AlbertTokenizer.from_pretrained(
    #         config.model_name, do_lower_case=config.model_name.endswith("uncased")
    #     )
    #     model = tx.AlbertModel.from_pretrained(config.model_name)
    # else:
    #     tokenizer = tx.AutoTokenizer.from_pretrained(
    #         config.model_name, do_lower_case=config.model_name.endswith("uncased")
    #     )
    #     model = tx.AutoModel.from_pretrained(config.model_name)

    tokenizer = tx.AutoTokenizer.from_pretrained(
        config.model_name,
        do_lower_case="uncased" in config.model_name,
        is_split_into_words=True
    )
    model = CodeRearranger(tx.AutoModel.from_pretrained(config.model_name))
    model = DataParallel(model, device_ids=[0, 1, 2, 3])
    
    return tokenizer, model.to(config.device)
