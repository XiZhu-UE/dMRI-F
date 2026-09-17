"""Runtime setup shared by training and inference."""
import random
import numpy as np
import torch


def loader_kwargs(settings):
    options = dict(settings)
    if options['num_workers'] == 0:
        options.pop('prefetch_factor', None)
        options.pop('persistent_workers', None)
    return options


def resolve_device(name, cpu_fallback=False):
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            if cpu_fallback:
                return torch.device('cpu')
            raise RuntimeError(f'Configured device {name} requires CUDA')
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(f'Configured device {name} is unavailable; update the device in the corresponding pretraining.ymal or inference.ymal')
    return device


def seed_everything(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
