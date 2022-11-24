import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from ppocr.modeling.heads.rec_crf_head import CrfHead


class CTCHead(nn.Module):
    def __init__(self, in_channels, out_channels=6625, fc_decay=0.0004, **kwargs):
        super(CTCHead, self).__init__()
        self.fc = nn.Linear(
            in_channels,
            out_channels,
            bias=True,)
        self.out_channels = out_channels
        # self.crf_head = CrfHead(self.out_channels)

    def forward(self, x, labels=None):
        predicts = self.fc(x)
        # predicts = self.crf_head(predicts)
        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            return predicts
        return predicts,x