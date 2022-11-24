# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


class CTCLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(reduction='mean',zero_infinity=True)
        # 预训练，weight取值较小
        self.weight = 0.00006
        self.kldiv = nn.KLDivLoss(reduction='batchmean')


    def forward(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[0]
        predicts = predicts.permute(1, 0, 2)
        self.num_classes = predicts.shape[-1]
        N, B, _ = predicts.shape

        preds_lengths = torch.LongTensor([N] * B)


        predicts = torch.nn.functional.log_softmax(predicts, dim=2)

        # preds_lengths = torch.Tensor([N] * B, dtype='int64')
        labels = batch[1]
        label_lengths = batch[2]
        # log_probs  # shape(max_len,batch_size,char_len)
        ctc_loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        kl_inp = predicts.transpose(0, 1)
        kl_tar = torch.full_like(kl_inp, 1. / self.num_classes)
        kldiv_loss = self.kldiv(kl_inp, kl_tar)
        # alpha=0.25
        # gamma=0.5
        # p = torch.exp(-ctc_loss)
        # focal_ctc_loss = ((alpha) * ((1 - p) ** gamma) * (ctc_loss))
        # loss = focal_ctc_loss
        # print(ctc_loss, kldiv_loss)
        loss = (1. - self.weight) * ctc_loss + self.weight * kldiv_loss
        # loss = ctc_loss
        return {'loss': loss}
