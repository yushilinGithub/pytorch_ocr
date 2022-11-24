# -*- coding: utf-8 -*-

# File   : rec_crf_head.py
# Date   : 2021-09-27
# Author : kaixiang
# Description:cnn+lstm+fc+crf
import torch.nn as nn
import torch
from torch import autograd


def argmax(vec):
    # return the argmax as a python int
    # 获取该维度上最大值的下标索引
    _, idx = torch.max(vec, 1)
    return idx.item()

# 前项算法
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class CrfHead(nn.Module):
    def __init__(self,tagset_size):
        super(CrfHead,self).__init__()
        self.tagset_size = tagset_size

        # 转移约束矩阵,0为blank,blank之间的标签转移忽略
        # self.transitions.data[0:] = 1
        # 转移矩阵,定义转移矩阵
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))


    def forward(self,launch_matrix):
        """
        # 前向传播
        """
        for index ,launch_matrix_single in enumerate(launch_matrix):
            # lstm_out --(max_len,target_size)
            forward_matrix = self.forward_step(launch_matrix_single)
            launch_matrix[index] = forward_matrix
        # return loss_tensor.view(out.size(0),-1)
        return launch_matrix


    # 前项算法
    def forward_step(self, feats):
        """
        每一个时间步在一个完整路径的得分
        :param feats:
        :return:
        """
        # 初始化单个时间步对应每一个标签的标签值
        # init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # 将参数进行梯度更新
        new_feats = feats.detach()
        # forward_var = autograd.Variable(init_alphas)
        for index,feat in enumerate(feats):
            pre_index = feat.argmax()  # 直接取argmax，优化argmax解码方式
            feat_trans = feat+self.transitions[pre_index]  # 相当于发射概率，广播是相同的

            new_feats[index] = feat_trans
        # 得到该时间步的加权
        return new_feats




