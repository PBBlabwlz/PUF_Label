import json
import math
import os
import random
from dataclasses import dataclass, field
import itertools

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import imageio

import dmp
from dmp.utils.config import parse_structured
from dmp.utils.typing import *


def _parse_scene_list(scene_path):
    train_path = os.path.join(scene_path, 'train_data')
    results = sorted([file for file in os.listdir(train_path) if file.endswith('_1.npy')])

    return results


@dataclass
class PUFDataModuleConfig:
    scene_list: Any = ""
    eval_scene_list: Any = ""

    repeat: int = 1

    batch_size: int = 1
    eval_batch_size: int = 1
    num_workers: int = 16

    same_p: float = 0.2
    quantize_number: int = 9
    min_number: int = -3
    max_number: int = 5

    train_triplet: bool = False
    test_full_data: bool = False


class PUFDataset(Dataset):
    def __init__(self, cfg: Any, split: str = "train") -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.cfg: PUFDataModuleConfig = cfg
        self.split = split
        self.repeat = self.cfg.repeat

        if self.split == "train":
            self.all_datas = _parse_scene_list(self.cfg.scene_list)
            self.all_datas = self.all_datas * self.repeat
            self.len = len(self.all_datas)

        elif self.split == "val":
            self.sample_path = os.path.join(self.cfg.eval_scene_list, 'test_sample')
            self.ref_path = os.path.join(self.cfg.eval_scene_list, 'test_ref')
            self.results_sample = sorted([file for file in os.listdir(self.sample_path) if file.endswith('.npy')])
            self.results_ref = sorted([file for file in os.listdir(self.ref_path) if file.endswith('.npy')])
            self.paired_ref_sample_results = list(itertools.product(self.results_ref, self.results_sample))
            self.len = len(self.paired_ref_sample_results)

        elif self.split == "test":
            if self.cfg.test_full_data:
                self.sample_path = os.path.join(self.cfg.eval_scene_list, 'full_data')
                self.ref_path = self.sample_path
                self.results_sample = sorted([file for file in os.listdir(self.sample_path) if file.endswith('2.npy')])
                self.results_ref = sorted([file for file in os.listdir(self.ref_path) if file.endswith('1.npy')])
            else:
                self.sample_path = os.path.join(self.cfg.eval_scene_list, 'test_sample')
                self.ref_path = os.path.join(self.cfg.eval_scene_list, 'test_ref')
                self.results_sample = sorted([file for file in os.listdir(self.sample_path) if file.endswith('.npy')])
                self.results_ref = sorted([file for file in os.listdir(self.ref_path) if file.endswith('.npy')])
            self.paired_ref_sample_results = list(itertools.product(self.results_ref, self.results_sample))
            self.len = len(self.paired_ref_sample_results)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        try:
            return self.try_get_item(index)
        except Exception as e:
            print(f"Failed to load {index}: {e}")
            return None

    def try_get_item(self, index):
        if self.split == "train":
            if self.cfg.train_triplet:
                out = self.get_triplet_sample(index)
            else:
                out = self.get_train_sample(index)
        else:
            out = self.get_eval_sample(index)

        return out

    def get_triplet_sample(self, index):
        train_path = os.path.join(self.cfg.scene_list, 'train_data')

        file_name = self.all_datas[index]
        number_anchor = file_name.split(os.sep)[-1].split('_')[0]
        path_anchor = os.path.join(train_path, number_anchor + '_1.npy')
        path_positive = os.path.join(train_path, number_anchor + '_2.npy')
        # random sample an index!=index
        index_negative = random.choice([i for i in range(self.len) if i != index])
        file_name_negative = self.all_datas[index_negative]
        number_negative = file_name_negative.split(os.sep)[-1].split('_')[0]
        path_negative = os.path.join(train_path, number_negative + '_2.npy')

        sample_anchor, _ = self.np_loader(path_anchor)
        sample_positive, _ = self.np_loader(path_positive)
        sample_negative, _ = self.np_loader(path_negative)

        out = {
            "sample_anchor": sample_anchor,
            "sample_positive": sample_positive,
            "sample_negative": sample_negative,
            "number_anchor": number_anchor,
            "number_positive": number_anchor,
            "number_negative": number_negative,
        }
        return out

    def get_train_sample(self, index):
        train_path = os.path.join(self.cfg.scene_list, 'train_data')

        if random.random() < self.cfg.same_p:
            file_name = self.all_datas[index]
            number = file_name.split(os.sep)[-1].split('_')[0]
            number_1 = number
            number_2 = number
            label = torch.tensor(0)
        else:
            index_1, index_2 = random.sample(range(0, self.len), 2)
            file_name_1, file_name_2 = self.all_datas[index_1], self.all_datas[index_2]
            number_1 = file_name_1.split(os.sep)[-1].split('_')[0]
            number_2 = file_name_2.split(os.sep)[-1].split('_')[0]
            label = torch.tensor(1)

        path_1 = os.path.join(train_path, number_1 + '_1.npy')
        path_2 = os.path.join(train_path, number_2 + '_2.npy')
        sample_1, _ = self.np_loader(path_1)
        sample_2, _ = self.np_loader(path_2)

        out = {
            "sample_1": sample_1,
            "sample_2": sample_2,
            "label": label,
            "number_1": number_1,
            "number_2": number_2,
        }
        return out

    def get_eval_sample(self, index):
        pair_data = self.paired_ref_sample_results[index]
        file_name_1, file_name_2 = pair_data
        number_1 = file_name_1.split(os.sep)[-1].split('_')[0]
        number_2 = file_name_2.split(os.sep)[-1].split('_')[0]
        path_1 = os.path.join(self.ref_path, file_name_1)
        path_2 = os.path.join(self.sample_path, file_name_2)
        sample_1, original_sample_1 = self.np_loader(path_1)
        sample_2, original_sample_2 = self.np_loader(path_2)

        if number_1 == number_2:
            label = torch.tensor(0)
        else:
            label = torch.tensor(1)

        out = {
            "sample_1": sample_1,
            "sample_2": sample_2,
            "label": label,
            "number_1": number_1,
            "number_2": number_2,
            "original_sample_1": original_sample_1,
            "original_sample_2": original_sample_2,
        }
        return out

    def np_loader(self, path):
        x_origin = np.load(path)
        x = torch.from_numpy(x_origin)
        x = (x.float() - self.cfg.min_number) / self.cfg.quantize_number
        x = 2 * x - 1

        return x, x_origin

    def collate(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        batch = torch.utils.data.default_collate(batch)

        return batch


class PUFDataModule(pl.LightningDataModule):
    cfg: PUFDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(PUFDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = PUFDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = PUFDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = PUFDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.val_dataset.collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.test_dataset.collate,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    conf = {
        "scene_list": "",
        "eval_scene_list": "",
        "batch_size": 4,
        "num_workers": 0,
    }

    dataset = PUFDataModule(conf)
    dataset.setup()
    # dataloader = dataset.train_dataloader()
    dataloader = dataset.test_dataloader()
    for batch in dataloader:
        breakpoint()

    breakpoint()
