import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import dmp


def reset_ema_weights(state_dict):
    new_state_dict = {}

    ema_keys_set = []
    for key, value in state_dict.items():
        assert key.startswith("backbone.") or key.startswith("backbone_ema.")
        if key.startswith("backbone.") and not key.startswith("backbone_ema"):
            new_state_dict[key] = value 
            ema_key = "backbone_ema." + key[len("backbone."):].replace(".", "")
            assert ema_key in state_dict
            ema_keys_set.append(ema_key)
            new_state_dict[ema_key] = value

    for key, value in state_dict.items():
        if key.startswith("backbone_ema."):
            # assert key in ema_keys_set, key
            if key not in ema_keys_set:
                dmp.warn(f"key {key} not in ema_keys_set")

    dmp.warn("restart ema!!!")

    return new_state_dict


def get_scheduler(name):
    if hasattr(lr_scheduler, name):
        return getattr(lr_scheduler, name)
    else:
        raise NotImplementedError


def getattr_recursive(m, attr):
    for name in attr.split("."):
        m = getattr(m, name)
    return m


def get_parameters(model, name):
    module = getattr_recursive(model, name)
    if isinstance(module, nn.Module):
        return module.parameters()
    elif isinstance(module, nn.Parameter):
        return module
    return []


def parse_optimizer(config, model, params=None):
    if params is None:
        if hasattr(config, "params"):
            params = [
                {"params": get_parameters(model, name), "name": name, **args}
                for name, args in config.params.items()
            ]
            dmp.debug(f"Specify optimizer params: {config.params}")
        else:
            params = model.parameters()
    if config.name in ["FusedAdam"]:
        import apex

        optim = getattr(apex.optimizers, config.name)(params, **config.args)
    elif config.name in ["Adam8bit", "AdamW8bit"]:
        import bitsandbytes as bnb

        optim = bnb.optim.Adam8bit(params, **config.args)
    else:
        optim = getattr(torch.optim, config.name)(params, **config.args)
    return optim


def parse_scheduler_to_instance(config, optimizer):
    if config.name == "ChainedScheduler":
        schedulers = [
            parse_scheduler_to_instance(conf, optimizer) for conf in config.schedulers
        ]
        scheduler = lr_scheduler.ChainedScheduler(schedulers)
    elif config.name == "Sequential":
        schedulers = [
            parse_scheduler_to_instance(conf, optimizer) for conf in config.schedulers
        ]
        scheduler = lr_scheduler.SequentialLR(
            optimizer, schedulers, milestones=config.milestones
        )
    else:
        scheduler = getattr(lr_scheduler, config.name)(optimizer, **config.args)
    return scheduler


def parse_scheduler(config, optimizer):
    interval = config.get("interval", "epoch")
    assert interval in ["epoch", "step"]
    if config.name == "SequentialLR":
        scheduler = {
            "scheduler": lr_scheduler.SequentialLR(
                optimizer,
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ],
                milestones=config.milestones,
            ),
            "interval": interval,
        }
    elif config.name == "ChainedScheduler":
        scheduler = {
            "scheduler": lr_scheduler.ChainedScheduler(
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ]
            ),
            "interval": interval,
        }
    else:
        scheduler = {
            "scheduler": get_scheduler(config.name)(optimizer, **config.args),
            "interval": interval,
        }
    return scheduler
