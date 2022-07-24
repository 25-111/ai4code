import torch
import torch.nn as nn
from torch.nn import DataParallel

import transformers as tx


def get_model(config):
    model = NotebookArranger(tx.AutoModel.from_pretrained(config.model_path))

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
    return model.to(config.device)


class NotebookArranger(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.model = pretrained_model
        self.fc = nn.Linear(769, 1)

    def forward(self, ids, mask, fts):
        x = self.model(ids, mask)[0]
        y = self.fc(torch.cat((x[:, 0, :], fts), 1))
        return y
