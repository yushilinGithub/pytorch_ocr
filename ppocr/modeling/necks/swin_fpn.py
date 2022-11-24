# -*- coding: utf-8 -*-

# File   : swin_fpn.py
# Date   : 2022-02-14
# Author : kaixiang
# Description:

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class SWINFPN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(SWINFPN, self).__init__()
        self.out_channels = out_channels

        self.in3_conv = nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False)
        self.in4_conv = nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False)
        self.in5_conv = nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False)
        self.p5_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False)
        self.p4_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False)
        self.p3_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False)


    def forward(self, x):
        c3, c4, c5 = x

        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)

        out4 = in4 + F.interpolate(
            in5, scale_factor=2, mode="nearest", )  # align_mode=1)  # 1/16
        out3 = in3 + F.interpolate(
            out4, scale_factor=2, mode="nearest", )  # align_mode=1)  # 1/8

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p5 = F.interpolate(p5, scale_factor=4, mode="nearest", )  # align_mode=1)
        p4 = F.interpolate(p4, scale_factor=2, mode="nearest", )  # align_mode=1)

        fuse = torch.cat([p5, p4, p3, p4], dim=1)  # scale (4,256,320,240)
        return fuse
