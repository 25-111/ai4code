import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import transformers


def get_model(config):
    if config.model_name == "bert-large":
        model_config = BertLargeConfig()
    elif config.model_name == "distill-beart":
        model_config = DistillBertConfig()
    return NotebookModel(model_config)


class NotebookModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, 1)

    def forward(self, ids, mask):  # ? Distill Bert
        x = self.config.model(ids, mask)[0]
        x = self.top(x[:, 0, :])
        return x

    def forward(self, ids, mask, token_type_ids):  # ? Bert Large
        _, x = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        x = self.fc(self.drop(x))
        return x


class BertLargeConfig:
    model_name = "bert-large-uncased"
    tokenizer = transformers.BertTokenizer.from_pretrained(
        model_name, do_lower_case=True
    )
    model = transformers.BertModel.from_pretrained(model_name)
    # scaler = GradScaler()


class DistillBertConfig:
    model_path = "../input/huggingface-bert-variants/distilbert-base-uncased/distilbert-base-uncased"
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(
        model_path, do_lower_case=True
    )
    model = transformers.DistilBertModel.from_pretrained(model_path)
