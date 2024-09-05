from dataclasses import dataclass, field
import math
import random
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import profiler
from scipy.spatial import distance

import dmp
from dmp.utils.misc import get_device
from dmp.systems.base import BaseLossConfig, BaseSystem
from dmp.utils.typing import *
from dmp.utils.misc import time_recorder as tr
from dmp.utils.lit_ema import LitEma


class DeepMetricSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        loss_cls: str = "dmp.systems.losses.FocalCrossEntropyLoss"
        loss: dict = field(default_factory=dict)

        use_ema: bool = True
        ema_decay: float = 0.9999
        test_with_ema: bool = True

    cfg: Config

    def configure(self):
        super().configure()
        self.backbone = dmp.find(self.cfg.backbone_cls)(self.cfg.backbone)
        self.use_ema = self.cfg.use_ema
        self.ema_decay = self.cfg.ema_decay
        self.test_with_ema = self.cfg.test_with_ema
        if self.use_ema:
            self.backbone_ema = LitEma(self.backbone, decay=self.ema_decay)
            dmp.info(f"Keeping EMAs of {len(list(self.backbone_ema.buffers()))}.")

        self.loss_fn = dmp.find(self.cfg.loss_cls)(self.cfg.loss)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.backbone_ema(self.backbone)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.backbone_ema.store(self.backbone.parameters())
            self.backbone_ema.copy_to(self.backbone)
            # if context is not None:
            #     dmp.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.backbone_ema.restore(self.backbone.parameters())
                # if context is not None:
                #     dmp.info(f"{context}: Restored training weights")

    def forward(self,
                data,
                ):
        feature = self.backbone(data)

        return feature

    def try_training_step(self, batch, batch_idx):
        device = get_device()
        sample_1 = batch["sample_1"].to(device)
        sample_2 = batch["sample_2"].to(device)
        label = batch["label"].to(device)

        feature_1 = self(sample_1)
        feature_2 = self(sample_2)
        sim = self.dense_similarity(feature_1, feature_2)

        loss = self.loss_fn(sim, label)

        outputs = {}
        self.check_train(batch, outputs,)

        return {"loss": loss, "out": outputs}

    def dense_similarity(self, feat_x, feat_y):
        sim = F.cosine_similarity(feat_x, feat_y, dim=1)
        sim = torch.mean(sim, dim=[1, 2])
        sim.clamp_(1e-8, 1 - 1e-8)
        return sim

    def on_check_train(self, batch, outputs):

        if (
                self.global_rank == 0
                and self.cfg.check_train_every_n_steps > 0
                and self.true_global_step % (self.cfg.check_train_every_n_steps*10) == 0
        ):
            pass

    def on_validation_start(self):
        self.test_results = {}
        self.cached_ref_representation = {}
        self.cached_sample_representation = {}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.test_step(batch, batch_idx)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if batch is None:
            dmp.info("Received None batch, skipping.")
            return None

        if self.use_ema and self.test_with_ema:
            with self.ema_scope("Validation with ema weights"):
                outputs = self.test_pipeline(batch)
        else:
            outputs = self.test_pipeline(batch)

        original_sample_1 = outputs["original_sample_1"].detach().cpu().view(-1)
        original_sample_2 = outputs["original_sample_2"].detach().cpu().view(-1)
        deep_representation_1 = outputs["deep_representation_1"]
        deep_representation_2 = outputs["deep_representation_2"]

        learned_similarity = self.dense_similarity(deep_representation_1, deep_representation_2).view(-1).detach().cpu().numpy()[0]
        hamming_distance = distance.hamming(original_sample_1, original_sample_2)
        hamming_similarity = 1 - hamming_distance

        self.test_results[outputs["pair_key"]] = {
            "learned_similarity": learned_similarity,
            #"learned_similarity": hamming_similarity,
            "hamming_similarity": hamming_similarity,
            "true_similarity": 1 - outputs["label"],
        }

    def on_test_start(self):
        self.test_results = {}
        self.cached_ref_representation = {}
        self.cached_sample_representation = {}

    def test_pipeline(self, batch):
        device = get_device()

        sample_1 = batch["sample_1"].to(device)
        sample_2 = batch["sample_2"].to(device)
        number_1 = batch["number_1"]
        number_2 = batch["number_2"]

        assert len(number_1) == 1
        assert len(number_2) == 1
        number_1 = number_1[0]
        number_2 = number_2[0]

        if number_1 in self.cached_ref_representation.keys():
            deep_representation_1 = self.cached_ref_representation[number_1]
        else:
            deep_representation_1 = self(sample_1)
            self.cached_ref_representation[number_1] = deep_representation_1

        if number_2 in self.cached_sample_representation.keys():
            deep_representation_2 = self.cached_sample_representation[number_2]
        else:
            deep_representation_2 = self(sample_2)
            self.cached_sample_representation[number_2] = deep_representation_2

        pair_key = f"{number_1}_{number_2}"
        outputs = {
            "pair_key": pair_key,
            "deep_representation_1": deep_representation_1,
            "deep_representation_2": deep_representation_2,
            "label": batch["label"],
            "original_sample_1": batch["original_sample_1"],
            "original_sample_2": batch["original_sample_2"],
        }


        return outputs

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def on_validation_epoch_end(self):
        sheet_dict = {}
        y_true_dict = {}
        y_pred_dict = {}
        ref_ids = list(self.cached_ref_representation.keys())
        sample_ids = list(self.cached_sample_representation.keys())
        assert len(ref_ids) == len(sample_ids)
        class_num = len(ref_ids)

        metric_keys = [key for key in self.test_results[f"{ref_ids[0]}_{sample_ids[0]}"].keys() if key != "true_similarity"]
        for key in metric_keys:
            sheet_dict[key] = [[0] * class_num for _ in range(class_num)]
            y_true_dict[key] = []
            y_pred_dict[key] = []

        for i in range(class_num):
            for j in range(class_num):
                ref_id = ref_ids[i]
                sample_id = sample_ids[j]
                pair_key = f"{ref_id}_{sample_id}"
                pair_info = self.test_results[pair_key]
                for key in metric_keys:
                    sheet_dict[key][i][j] = pair_info[key]
                    y_true_dict[key].append(pair_info["true_similarity"].detach().cpu().numpy()[0])
                    y_pred_dict[key].append(pair_info[key])
        
        self.evaluate(sheet_dict, y_true_dict, y_pred_dict, save_subdir=f"{self.true_global_step}")
