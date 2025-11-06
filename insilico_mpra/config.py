# -*- coding: utf-8 -*-
"""
Configuration module for MPRA LegNet training and inference.
"""

import json
import torch.nn as nn
from dataclasses import dataclass, asdict, InitVar
from pathlib import Path
from typing import Optional, Tuple, List, Union


@dataclass
class TrainingConfig:
    """Configuration class for training MPRA LegNet models."""

    # Model architecture parameters
    stem_ch: int
    stem_ks: int
    ef_ks: int
    ef_block_sizes: List[int]
    resize_factor: int
    pool_sizes: List[int]

    # Data augmentation parameters
    reverse_augment: bool
    use_reverse_channel: bool
    use_shift: bool
    max_shift: Optional[Tuple[int, int]]

    # Training parameters
    max_lr: float
    weight_decay: float
    epoch_num: int
    train_batch_size: int
    valid_batch_size: int

    # System parameters
    model_dir: str
    data_path: str
    device: int
    seed: int
    num_workers: int

    # Internal parameter
    training: InitVar[bool] = True

    def __post_init__(self, training: bool):
        """Post-initialization validation and setup."""
        self.check_params()
        model_dir = Path(self.model_dir)
        if training:
            model_dir.mkdir(exist_ok=True, parents=True)
            self.dump()

    def check_params(self):
        """Validate configuration parameters."""
        # if Path(self.model_dir).exists():
        #     print(f"Warning: model dir already exists: {self.model_dir}")
        if not self.reverse_augment:
            if self.use_reverse_channel:
                raise Exception("If model uses reverse channel, reverse augmentation must be performed")

    def dump(self, path: Optional[Union[str, Path]] = None):
        """Save configuration to JSON file."""
        if path is None:
            path = Path(self.model_dir) / "config.json"
        self.to_json(path)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, path: Union[str, Path]):
        """Save configuration to JSON file."""
        dt = self.to_dict()
        with open(path, 'w') as out:
            json.dump(dt, out, indent=4)

    @classmethod
    def from_dict(cls, dt: dict) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        return cls(**dt)

    @classmethod
    def from_json(cls, path: Union[Path, str], training: bool = False) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as inp:
            dt = json.load(inp)
        dt['training'] = training
        return cls.from_dict(dt)

    @property
    def in_ch(self) -> int:
        """Calculate number of input channels."""
        return 4 + self.use_reverse_channel

    def get_model(self) -> nn.Module:
        """Create and return model instance."""
        from .models.legnet import LegNet
        return LegNet(
            in_ch=self.in_ch,
            stem_ch=self.stem_ch,
            stem_ks=self.stem_ks,
            ef_ks=self.ef_ks,
            ef_block_sizes=self.ef_block_sizes,
            resize_factor=self.resize_factor,
            pool_sizes=self.pool_sizes
        )


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig(
        stem_ch=64,
        stem_ks=11,
        ef_ks=9,
        ef_block_sizes=[80, 96, 112, 128],
        resize_factor=4,
        pool_sizes=[2, 2, 2, 2],
        reverse_augment=True,
        use_reverse_channel=False,
        use_shift=True,
        max_shift=None,
        max_lr=0.01,
        weight_decay=0.1,
        model_dir="./models/default_model",
        data_path="../datasets/lenti_MPRA_K562_data.h5",
        epoch_num=25,
        device=0,
        seed=777,
        train_batch_size=1024,
        valid_batch_size=1024,
        num_workers=8,
        training=True
    )
