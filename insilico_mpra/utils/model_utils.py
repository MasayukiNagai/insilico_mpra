# -*- coding: utf-8 -*-
"""
Model utility functions for training and initialization.
"""

import torch
import torch.nn as nn
import random
import numpy as np
import math


def parameter_count(model):
    """Count total number of trainable parameters in model."""
    return sum(torch.prod(torch.tensor(p.shape)) for _, p in model.named_parameters())


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_weights(m):
    """Initialize model weights using appropriate strategies for each layer type."""
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0) 