from typing import *

import torch

module_dict: Dict[str, torch.nn.Module] = dict()


def module_path(module_name: str) -> str:
    return "log/model/{}.pt".format(module_name)


def register_module(module_name: str, module: torch.nn.Module):
    assert module_name not in module_dict
    module_dict[module_name] = module
    try:
        module.load_state_dict(torch.load(module_path(module_name)))
    except IOError:
        pass


def save_module(module_name: str):
    torch.save(module_dict[module_name].state_dict(), module_path(module_name))


def get_module(module_name: str):
    return module_dict[module_name]
