import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import transformers as tx


def get_model(config):
    if config.model_name.startswith("bert"):
        model_config = BertConfig(config)
    elif config.model_name.startswith("distill-beart"):
        model_config = DistillBertConfig(config)
    elif config.model_name.startswith("roberta"):
        model_config = RobertaConfig(config)
    elif config.model_name.startswith("albert"):
        model_config = AlbertConfig(config)
    else:
        model_config = OtherConfig(config)
    return CodeRearranger(model_config)


class CodeRearranger(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, 1)

    def forward(self, ids, mask):
        x = self.config.model(ids, mask)[0]
        x = self.top(x[:, 0, :])
        return x


# TODO: rename model_name to model_path for training on aze server
class BertConfig:
    def __init__(self, config):
        self.tokenizer = tx.BertTokenizer.from_pretrained(
            config.model_name, do_lower_case=config.model_name.endswith("uncased")
        )
        self.model = tx.BertModel.from_pretrained(config.model_name)
        # scaler = GradScaler()


class DistillBertConfig:
    def __init__(self, config):
        self.tokenizer = tx.DistilBertTokenizer.from_pretrained(
            config.model_name, do_lower_case=config.model_name.endswith("uncased")
        )
        self.model = tx.DistilBertModel.from_pretrained(config.model_name)


class RobertaConfig:
    def __init__(self, config):
        self.tokenizer = tx.RobertaTokenizer.from_pretrained(
            config.model_name, do_lower_case=config.model_name.endswith("uncased")
        )
        self.model = tx.RobertaModel.from_pretrained(config.model_name)


class AlbertConfig:
    def __init__(self, config):
        self.tokenizer = tx.AlbertTokenizer.from_pretrained(
            config.model_name, do_lower_case=config.model_name.endswith("uncased")
        )
        self.model = tx.AlbertModel.from_pretrained(config.model_name)


class OtherConfig:
    def __init__(self, config):
        self.tokenizer = tx.AutoTokenizer.from_pretrained(
            config.model_name, do_lower_case=config.model_name.endswith("uncased")
        )
        self.model = tx.AutoModel.from_pretrained(config.model_name)
