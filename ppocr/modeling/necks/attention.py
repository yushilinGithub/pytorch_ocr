# -*- coding: utf-8 -*-

# File   : attention.py
# Date   : 2021-09-28
# Author : kaixiang
# Description: Luong attention
import torch
import torch.nn.functional as F
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self,hidden_state,output):
        super(Attention,self).__init__()
        self.nn_hidden = nn.Linear(hidden_state, output,bias=True)

    def forward(self, hidden_state,output):
        """
        # 根据hidden_state和output 获取模式下权重
        :param hidden_state: batch_size hidden_size
        :param output: batch_size max_len hidden_size*bi
        :return:
        """
        """
        weight权重模式方法：
        general: 对hidden_state进行线性回归，然后和output进行相乘得到attention_weight
        """
        # hidden_state+BN 防止梯度爆炸
        nn_hidden = self.nn_hidden(hidden_state) # ==>batch_size,hidden_size*bi
        nn_hidden = F.gelu(nn_hidden)
        hidden = nn_hidden.unsqueeze(-1) # ==> batch_size,hidden_size*bi,1
        attn_factor = torch.bmm(output,hidden) # batch_size,max_len,1
        attn_factor = attn_factor.squeeze(-1) # ==> batch_size,max_len
        attn_weight = F.softmax(attn_factor,dim=-1)

        # 对输入input进行注意力权重配置----max_len
        return attn_weight  # batch_size,max_len

