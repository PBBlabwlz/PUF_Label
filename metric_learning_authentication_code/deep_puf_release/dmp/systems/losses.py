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
from dmp.utils.base import BaseObject
from dmp.utils.typing import *


class CrossEntropyLoss(BaseObject):
    def configure(self) -> None:
        super().configure()
        self.loss = nn.CrossEntropyLoss()

    def __call__(
            self,
            outputs,
            label: Float[Tensor, "B"],
    ):
        loss = self.loss(outputs, label.long())

        return loss


class FocalCrossEntropyLoss(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        gamma: float = 2.0

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.gamma = self.cfg.gamma

    def __call__(
            self,
            sim: Float[Tensor, "B"],
            label: Float[Tensor, "B"],
    ):
        w_1 = (1 - sim).detach() ** self.gamma
        w_2 = sim.detach() ** self.gamma

        loss = - (w_1 * torch.log(sim) * (1 - label.float()) + w_2 * torch.log(1 - sim) * label.float())
        loss = loss.mean()

        return loss

class MarginLoss(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        margin: float = 0.5

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.margin = self.cfg.margin

    def __call__(
            self,
            sim: Float[Tensor, "B"],
            label: Float[Tensor, "B"],
    ):
        pair_label = 1 - label.float()
        distance = 1-sim

        loss = pair_label * distance**2 + (1-pair_label) * torch.max(torch.zeros_like(distance), self.margin - distance)**2
        loss = loss.mean()

        return loss

class TripletLoss(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        margin: float = 0.5

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.margin = self.cfg.margin

    def __call__(
            self,
            sim_positive: Float[Tensor, "B"],
            sim_negative: Float[Tensor, "B"],
    ):

        distance_positive = 1-sim_positive
        distance_negative = 1-sim_negative

        loss = torch.max(torch.zeros_like(distance_positive), distance_positive - distance_negative + self.margin)
        loss = loss.mean()

        return loss
