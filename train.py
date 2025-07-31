#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for MPRA LegNet model.

Usage:
    python train.py --config config.json
    python train.py --data_path /path/to/data.h5 --model_dir ./models/my_model
"""

import argparse
import torch
import lightning.pytorch as pl
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from mpralegnet.config import TrainingConfig, get_default_config
from mpralegnet.models.lightning_module import LitModel
from mpralegnet.data.utils import create_dataloaders, check_h5_dataset
from mpralegnet.utils.model_utils import set_global_seed, parameter_count


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MPRA LegNet model')

    # Configuration options
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--data_path', type=str, help='Path to HDF5 data file')
    parser.add_argument('--model_dir', type=str, help='Directory to save model')
    parser.add_argument('--data_format', type=str, default='h5', choices=['h5', 'tsv'],
                        help='Data format (h5 or tsv)')

    # Training parameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, help='Maximum learning rate')
    parser.add_argument('--device', type=int, help='GPU device ID')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers')

    # Model parameters
    parser.add_argument('--stem_ch', type=int, help='Stem channels')
    parser.add_argument('--no_reverse_augment', action='store_true',
                        help='Disable reverse complement augmentation')
    parser.add_argument('--no_shift', action='store_true', help='Disable shift augmentation')

    # Debugging and inspection
    parser.add_argument('--check_data', action='store_true', help='Check data structure and exit')
    parser.add_argument('--dry_run', action='store_true', help='Setup model and data without training')

    return parser.parse_args()


def setup_config(args):
    """Setup configuration from arguments."""
    if args.config:
        # Load from configuration file
        config = TrainingConfig.from_json(args.config, training=True)
        print(f"Loaded configuration from {args.config}")
    else:
        # Use default configuration
        config = get_default_config()
        print("Using default configuration")

    # Override with command line arguments
    if args.data_path:
        config.data_path = args.data_path
    if args.model_dir:
        config.model_dir = args.model_dir
    if args.epochs:
        config.epoch_num = args.epochs
    if args.batch_size:
        config.train_batch_size = args.batch_size
        config.valid_batch_size = args.batch_size
    if args.learning_rate:
        config.max_lr = args.learning_rate
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.stem_ch:
        config.stem_ch = args.stem_ch
    if args.no_reverse_augment:
        config.reverse_augment = False
    if args.no_shift:
        config.use_shift = False

    return config


def setup_callbacks(config):
    """Setup training callbacks."""
    model_dir = Path(config.model_dir)

    callbacks = [
        ModelCheckpoint(
            dirpath=model_dir,
            save_top_k=1,
            monitor="val_pearson",
            mode="max",
            filename="best_model-{epoch:02d}-{val_pearson:.3f}",
            save_last=True,
            save_weights_only=True,
        ),
        EarlyStopping(
            monitor="val_pearson",
            mode="max",
            patience=10,
            verbose=True
        ),
        LearningRateMonitor(
            logging_interval='step'
        )
    ]

    return callbacks


def train_model(config, data_file, data_format='h5'):
    """Train the model with given configuration."""
    # Set random seed
    print(f"Setting global seed: {config.seed}")
    set_global_seed(config.seed)

    # Set PyTorch precision
    torch.set_float32_matmul_precision('medium')

    # Create model
    model = LitModel(config)
    print(f"Model parameters: {parameter_count(model.model).item():,}")

    # Create data loaders
    train_dl, val_dl = create_dataloaders(config, data_file, data_format)
    print(f"Training samples: {len(train_dl.dataset):,}")
    print(f"Validation samples: {len(val_dl.dataset):,}")

    # Setup callbacks
    callbacks = setup_callbacks(config)

    # Create trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[config.device] if torch.cuda.is_available() else 1,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        max_epochs=config.epoch_num,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        default_root_dir=config.model_dir,
        log_every_n_steps=50,
        val_check_interval=0.5,  # Validate twice per epoch
        enable_progress_bar=False,
        enable_model_summary=True
    )

    # Print training information
    print(f"\nTraining Configuration:")
    print(f"  Data: {data_file}")
    print(f"  Model directory: {config.model_dir}")
    print(f"  Epochs: {config.epoch_num}")
    print(f"  Batch size: {config.train_batch_size}")
    print(f"  Learning rate: {config.max_lr}")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'} {config.device}")
    print(f"  Reverse augmentation: {config.reverse_augment}")
    print(f"  Shift augmentation: {config.use_shift}")
    print()

    # Train model
    trainer.fit(model, train_dl, val_dl)

    # Get best model path
    best_model_path = callbacks[0].best_model_path
    print(f"\nTraining completed!")
    print(f"Best model saved at: {best_model_path}")

    return model, trainer, best_model_path


def main():
    """Main training function."""
    args = parse_arguments()

    # Setup configuration
    config = setup_config(args)

    # Check data structure if requested
    if args.check_data:
        print("Checking data structure...")
        check_h5_dataset(config.data_path)
        return

    # Verify data file exists
    if not Path(config.data_path).exists():
        raise FileNotFoundError(f"Data file not found: {config.data_path}")

    # Print configuration
    print("Training Configuration:")
    print(f"  Data path: {config.data_path}")
    print(f"  Model directory: {config.model_dir}")
    print(f"  Data format: {args.data_format}")

    if args.dry_run:
        print("Dry run mode - setting up model and data without training...")

        # Setup model and data
        set_global_seed(config.seed)
        model = LitModel(config)
        train_dl, val_dl = create_dataloaders(config, config.data_path, args.data_format)

        print(f"Model parameters: {parameter_count(model.model).item():,}")
        print(f"Training samples: {len(train_dl.dataset):,}")
        print(f"Validation samples: {len(val_dl.dataset):,}")
        print("Dry run completed successfully!")
        return

    # Train model
    try:
        model, trainer, best_model_path = train_model(config, config.data_path, args.data_format)
        print("Training completed successfully!")

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
