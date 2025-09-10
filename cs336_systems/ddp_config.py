from random import random
from collections.abc import Iterable

import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class TrainConfig():

    batch_size = 32
    context_length = 256
    d_model = 512
    d_ff = 1344
    num_layers = 4
    num_heads = 8
    rope_theta = 10000
    vocab_size = 10000

    # train parameter
    max_seq_len = 64
    num_steps = 100
    learning_rate = 1e-3
    weight_decay = 1e-5
    gradient_clipping = 1.0
    epsilon = 1e-8
    beta1 = 0.9
    beta2 = 0.95
    warmup_steps = 0

    # data
    seed = 42
    # set_seed(seed)

    # ddp train
    ddp_backend = 'gloo'
    world_size = 2

    # log
    log_interval = 10


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def generate_random_data(batch_size, context_length, vocab_size, seed=None):
    if seed is not None:
        set_seed(seed)
    data = torch.randint(0, vocab_size, (context_length, batch_size))  # , dtype=torch.float32

    torch.save(data, 'random_data.pt')
    return data


def save_checkpoint(model, path:str="initial_param.pt"):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path:str="initial_param.pt"):
    model.load_state_dict(torch.load(path))


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return

    total_norm_sq = torch.zeros(1, device=params[0].grad.device)
    for p in params:
        total_norm_sq += p.grad.pow(2).sum()

    total_norm = torch.sqrt(total_norm_sq)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in params:
            p.grad.mul_(scale)
