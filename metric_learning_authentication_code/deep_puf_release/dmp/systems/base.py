import os
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from einops import rearrange
from torch.autograd import profiler

import dmp
from .utils import parse_optimizer, parse_scheduler, reset_ema_weights
from ..utils.base import (
    Updateable,
    update_end_if_possible,
    update_if_possible,
)
from ..utils.config import parse_structured
from ..utils.misc import C, cleanup, get_device, load_module_weights
from ..utils.saving import SaverMixin
from ..utils.typing import *
from dmp.utils.misc import time_recorder as tr


@dataclass
class BaseLossConfig:
    pass


class BaseSystem(pl.LightningModule, Updateable, SaverMixin):
    @dataclass
    class Config:
        loss: BaseLossConfig = BaseLossConfig()
        optimizer: dict = field(default_factory=dict)
        scheduler: Optional[dict] = None
        weights: Optional[str] = None
        weights_ignore_modules: Optional[List[str]] = None
        weights_mapping: Optional[List[Dict[str, str]]] = None
        check_train_every_n_steps: int = 0
        check_val_limit_rank: int = 8
        cleanup_after_validation_step: bool = False
        cleanup_after_test_step: bool = False
        load_restart_ema: bool = False

    cfg: Config

    def __init__(self, cfg, resumed=False) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self._save_dir: Optional[str] = None
        self._resumed: bool = resumed
        self._resumed_eval: bool = False
        self._resumed_eval_status: dict = {"global_step": 0, "current_epoch": 0}

        self.configure()
        if self.cfg.weights is not None:
            self.load_weights(
                self.cfg.weights,
                self.cfg.weights_ignore_modules,
                self.cfg.weights_mapping,
            )
        self.post_configure()

    def load_weights(
        self,
        weights: str,
        ignore_modules: Optional[List[str]] = None,
        mapping: Optional[List[Dict[str, str]]] = None,
    ):
        state_dict, epoch, global_step = load_module_weights(
            weights,
            ignore_modules=ignore_modules,
            mapping=mapping,
            map_location="cpu",
        )

        if self.cfg.load_restart_ema:
            state_dict = reset_ema_weights(state_dict)
        # state_dict = reset_ema_weights(state_dict)

        try:
            self.load_state_dict(state_dict, strict=False)
        except Exception as e:
            # 过滤掉不匹配的键
            model_state_dict = self.state_dict()

            for key in list(state_dict.keys()):
                if key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
                    dmp.warn(f"Skipping {key} due to size mismatch.")
                    del state_dict[key]
                elif key not in model_state_dict:
                    dmp.warn(f"Skipping {key} due to missing key.")
                    del state_dict[key]
            self.load_state_dict(state_dict, strict=False)
            dmp.warn(f"Failed to load some keys from the checkpoint: {e}")
        # restore step-dependent states
        self.do_update_step(epoch, global_step, on_load_weights=True)


    def set_resume_status(self, current_epoch: int, global_step: int):
        # restore correct epoch and global step in eval
        self._resumed_eval = True
        self._resumed_eval_status["current_epoch"] = current_epoch
        self._resumed_eval_status["global_step"] = global_step

    @property
    def resumed(self):
        # whether from resumed checkpoint
        return self._resumed

    @property
    def true_global_step(self):
        if self._resumed_eval:
            return self._resumed_eval_status["global_step"]
        else:
            return self.global_step

    @property
    def true_current_epoch(self):
        if self._resumed_eval:
            return self._resumed_eval_status["current_epoch"]
        else:
            return self.current_epoch

    def configure(self) -> None:
        pass

    def post_configure(self) -> None:
        """
        executed after weights are loaded
        """
        pass

    def C(self, value: Any) -> float:
        return C(value, self.true_current_epoch, self.true_global_step)

    def configure_optimizers(self):
        optim = parse_optimizer(self.cfg.optimizer, self)
        ret = {
            "optimizer": optim,
        }
        if self.cfg.scheduler is not None:
            ret.update(
                {
                    "lr_scheduler": parse_scheduler(self.cfg.scheduler, optim),
                }
            )
        return ret

    def on_fit_start(self) -> None:
        if self._save_dir is not None:
            dmp.info(f"Validation results will be saved to {self._save_dir}")
        else:
            dmp.warn(
                f"Saving directory not set for the system, visualization results will not be saved"
            )

    def training_step(self, batch, batch_idx):
        # 检查 batch 是否为 None

        if batch is None:
            dmp.info("Received None batch, skipping.")
            return None

        out = self.try_training_step(batch, batch_idx)

        return out

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        """Called to perform backward on the loss returned in :meth:`training_step`. Override this hook with your own
        implementation if you need to.

        Args:
            loss: The loss tensor returned by :meth:`training_step`. If gradient accumulation is used, the loss here
                holds the normalized value (scaled by 1 / accumulation steps).

        Example::

            def backward(self, loss):
                loss.backward()

        """
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        tr.start("backward")
        if self._fabric:
            self._fabric.backward(loss, *args, **kwargs)
        else:
            loss.backward(*args, **kwargs)
        tr.end("backward")
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    def try_training_step(self, batch, batch_idx):
        raise NotImplementedError

    def check_train(self, batch, outputs, **kwargs):
        if (
            self.global_rank == 0
            and self.cfg.check_train_every_n_steps > 0
            and self.true_global_step % self.cfg.check_train_every_n_steps == 0
        ):
            self.on_check_train(batch, outputs, **kwargs)

    def on_check_train(self, batch, outputs, **kwargs):
        pass

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_test_epoch_end(self):
        pass

    def on_test_end(self) -> None:
        if self._save_dir is not None:
            dmp.info(f"Test results saved to {self._save_dir}")

    def on_predict_start(self) -> None:
        pass

    def on_predict_end(self) -> None:
        if self._save_dir is not None:
            dmp.info(f"Export assets saved to {self._save_dir}")

    def preprocess_data(self, batch, stage):
        pass

    """
    Implementing on_after_batch_transfer of DataModule does the same.
    But on_after_batch_transfer does not support DP.
    """

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        tr.start("train_batch_start")
        self.preprocess_data(batch, "train")
        self.dataset = self.trainer.train_dataloader.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)
        tr.end("train_batch_start")

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "validation")
        self.dataset = self.trainer.val_dataloaders.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "test")
        self.dataset = self.trainer.test_dataloaders.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.preprocess_data(batch, "predict")
        self.dataset = self.trainer.predict_dataloaders.dataset
        update_if_possible(self.dataset, self.true_current_epoch, self.true_global_step)
        self.do_update_step(self.true_current_epoch, self.true_global_step)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        tr.start("train_batch_end")
        self.dataset = self.trainer.train_dataloader.dataset
        update_end_if_possible(
            self.dataset, self.true_current_epoch, self.true_global_step
        )
        self.do_update_step_end(self.true_current_epoch, self.true_global_step)
        tr.end("train_batch_end")

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.dataset = self.trainer.val_dataloaders.dataset
        update_end_if_possible(
            self.dataset, self.true_current_epoch, self.true_global_step
        )
        self.do_update_step_end(self.true_current_epoch, self.true_global_step)
        if self.cfg.cleanup_after_validation_step:
            # cleanup to save vram
            cleanup()

    def on_test_batch_end(self, outputs, batch, batch_idx):
        self.dataset = self.trainer.test_dataloaders.dataset
        update_end_if_possible(
            self.dataset, self.true_current_epoch, self.true_global_step
        )
        self.do_update_step_end(self.true_current_epoch, self.true_global_step)
        if self.cfg.cleanup_after_test_step:
            # cleanup to save vram
            cleanup()

    def on_predict_batch_end(self, outputs, batch, batch_idx):
        self.dataset = self.trainer.predict_dataloaders.dataset
        update_end_if_possible(
            self.dataset, self.true_current_epoch, self.true_global_step
        )
        self.do_update_step_end(self.true_current_epoch, self.true_global_step)
        if self.cfg.cleanup_after_test_step:
            # cleanup to save vram
            cleanup()

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        pass

    def on_before_optimizer_step(self, optimizer):
        """
        # some gradient-related debugging goes here, example:
        from lightning.pytorch.utilities import grad_norm
        norms = grad_norm(self.geometry, norm_type=2)
        print(norms)
        for name, p in self.named_parameters():
            if p.grad is None:
                dmp.info(f"{name} does not receive gradients!")
        """
        pass
