import torch
import torch.nn as nn
import torch.nn.functional as F


class MarkdownModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top = nn.Linear(768, 1)

    def forward(self, ids, mask):
        x = self.config.model(ids, mask)[0]
        x = self.top(x[:, 0, :])
        return x
