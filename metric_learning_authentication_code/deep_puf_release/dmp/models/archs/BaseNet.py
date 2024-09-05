import time
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import random
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import einops
from einops import rearrange

import dmp
from dmp.utils.misc import get_device
from dmp.utils.misc import time_recorder as tr
from dmp.utils.base import BaseModule
from dmp.utils.typing import *


class DMPNet(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channel: int = 3
        image_resolution: int = 32

    cfg: Config

    def configure(self) -> None:
        super().configure()
        in_channel = self.cfg.in_channel
        in_res = self.cfg.image_resolution
        mid_channel = in_channel * 16
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.dense_layer1 = RDB(in_channel=mid_channel, d_list=(1, 2, 1), inter_feature=mid_channel // 4)
        self.dense_layer2 = RDB(in_channel=mid_channel, d_list=(1, 2, 1), inter_feature=mid_channel // 4)
        self.dense_layer3 = RDB(in_channel=mid_channel, d_list=(1, 2, 1), inter_feature=mid_channel // 4)

        self.layer3 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer_norm = nn.LayerNorm((mid_channel, in_res, in_res), elementwise_affine=False)

    def forward(
            self,
            x: Float[Tensor, "B C H W"],
    ):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        x = self.dense_layer3(x)

        x = self.layer_norm(self.relu(self.layer3(x)))

        return x


class RDB(nn.Module):
    def __init__(self, in_channel, d_list, inter_feature):
        super(RDB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_feature, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_feature
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)

        t = self.conv_post(t)
        return t + x

class ident_layer(nn.Module):
    def __init__(self):
        super(ident_layer, self).__init__()

    def forward(self, x):
        return x

class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out

class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out