# -*- coding: utf-8 -*-
"""
MPRA LegNet: A deep learning model for Massively Parallel Reporter Assay (MPRA) data.

This package provides tools for training and inference with the LegNet architecture
for predicting regulatory activity from DNA sequences.
"""

__version__ = "1.0.0"
__author__ = "MPRA LegNet Team"

from .config import TrainingConfig
from .models.legnet import LegNet
from .models.lightning_module import LitModel

__all__ = [
    "TrainingConfig",
    "LegNet", 
    "LitModel"
] 