# -*- coding: utf-8 -*-
"""
Data utility functions for dataset creation and manipulation.
"""

import h5py
import pandas as pd
from torch.utils.data import DataLoader
from .datasets import HDF5Dataset, TSVDataset


def check_h5_dataset(file_path):
    """
    Check and display the structure of an HDF5 dataset.
    
    Args:
        file_path: Path to HDF5 file
    """
    try:
        with h5py.File(file_path, 'r') as f:
            print("Keys in the HDF5 file:")
            for key in f.keys():
                print(f" - {key}")
            print("\nDataset structure:")
            def print_structure(name, obj):
                if hasattr(obj, 'shape'):
                    print(f"{name}: {obj.shape} {obj.dtype}")
            f.visititems(print_structure)
    except Exception as e:
        print(f"An error occurred: {e}")


def create_dataloaders(config, data_file: str, data_format: str = 'h5'):
    """
    Create train/val dataloaders from either HDF5 or TSV format.
    
    Args:
        config: TrainingConfig object
        data_file: Path to data file
        data_format: Data format ('h5' or 'tsv')
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    if data_format == 'h5':
        train_ds = HDF5Dataset(
            data_file,
            split='train',
            use_reverse=config.reverse_augment,
            use_shift=config.use_shift,
            use_reverse_channel=config.use_reverse_channel,
            max_shift=config.max_shift,
            training=True
        )

        val_ds = HDF5Dataset(
            data_file,
            split='valid',
            use_reverse_channel=config.use_reverse_channel,
            training=False
        )

    elif data_format == 'tsv':
        # Legacy TSV support
        df = pd.read_csv(data_file, sep='\t')
        df.columns = ['seq_id', 'seq', 'mean_value', 'fold_num', 'rev'][:len(df.columns)]

        if "rev" in df.columns:
            df = df[df.rev == 0]

        # Simple train/val split for demonstration
        train_df = df.sample(frac=0.8, random_state=config.seed)
        val_df = df.drop(train_df.index)

        train_ds = TSVDataset(
            train_df.reset_index(drop=True),
            use_reverse=config.reverse_augment,
            use_shift=config.use_shift,
            use_reverse_channel=config.use_reverse_channel,
            max_shift=config.max_shift,
            training=True
        )

        val_ds = TSVDataset(
            val_df.reset_index(drop=True),
            use_reverse_channel=config.use_reverse_channel,
            training=False
        )
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

    train_dl = DataLoader(
        train_ds,
        batch_size=config.train_batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=config.valid_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )

    return train_dl, val_dl


def create_test_dataloader(config, data_file: str, data_format: str = 'h5', split: str = 'test'):
    """
    Create test dataloader.
    
    Args:
        config: TrainingConfig object
        data_file: Path to data file
        data_format: Data format ('h5' or 'tsv')
        split: Data split name
        
    Returns:
        Test dataloader
    """
    if data_format == 'h5':
        test_ds = HDF5Dataset(
            data_file,
            split=split,
            use_reverse_channel=config.use_reverse_channel,
            training=False
        )
    else:
        raise NotImplementedError("TSV test dataloader not implemented")

    test_dl = DataLoader(
        test_ds,
        batch_size=config.valid_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )

    return test_dl 