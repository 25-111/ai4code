import torch.nn as nn
from torch.cuda.amp import GradScaler
import transformers as tx


def get_model(config):
    # TODO: rename model_name to model_path for training on msj server
    if config.model_name.startswith("bert"):
        tokenizer = tx.BertTokenizer.from_pretrained(
            config.model_name, do_lower_case=config.model_name.endswith("uncased")
        )
        model = tx.BertModel.from_pretrained(config.model_name)
    elif config.model_name.startswith("distill-beart"):
        tokenizer = tx.DistilBertTokenizer.from_pretrained(
            config.model_name, do_lower_case=config.model_name.endswith("uncased")
        )
        model = tx.DistilBertModel.from_pretrained(config.model_name)
    elif config.model_name.startswith("roberta"):
        tokenizer = tx.RobertaTokenizer.from_pretrained(
            config.model_name, do_lower_case=config.model_name.endswith("uncased")
        )
        model = tx.RobertaModel.from_pretrained(config.model_name)
    elif config.model_name.startswith("albert"):
        tokenizer = tx.AlbertTokenizer.from_pretrained(
            config.model_name, do_lower_case=config.model_name.endswith("uncased")
        )
        model = tx.AlbertModel.from_pretrained(config.model_name)
    else:
        tokenizer = tx.AutoTokenizer.from_pretrained(
            config.model_name, do_lower_case=config.model_name.endswith("uncased")
        )
        model = tx.AutoModel.from_pretrained(config.model_name)
    return tokenizer, model


class CodeRearranger(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fc = nn.Linear(769, 1)

    def forward(self, ids, mask):
        x = self.model(ids, mask)[0]
        x = self.fc(x[:, 0, :])
        return x
