# -*- coding: utf-8 -*-
"""
Data utility functions for dataset creation and manipulation.
"""

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

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


def split_h5_dataset(
    in_path,
    out_path = None,
    ratios=(0.8, 0.1, 0.1),
    seed=None,
    x_key="x",
    y_key="y",
    compression='gzip'
):
    """Shuffle once, split by ratios, and write x_train/y_train/x_valid/y_valid/x_test/y_test."""
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("ratios must sum to 1")

    in_path = Path(in_path)
    if out_path is None:
        out_path = in_path.with_name(in_path.stem + "_split.h5")
    out_path = Path(out_path)

    rng = np.random.default_rng(seed)

    # Load everything into memory
    with h5py.File(in_path, "r") as fin:
        X = np.asarray(fin[x_key])  # shape: (N, ...)
        Y = np.asarray(fin[y_key])  # shape: (N, ...) or (N,)
    N = X.shape[0]
    if Y.shape[0] != N:
        raise ValueError("x and y must have the same first dimension")

    # Indices
    perm = rng.permutation(N)
    n_train = int(ratios[0] * N)
    n_valid = int(ratios[1] * N)
    # n_test  = N - n_train - n_valid

    i_train = perm[:n_train]
    i_valid = perm[n_train:n_train + n_valid]
    i_test  = perm[n_train + n_valid:]

    # Slice once per split
    X_tr, Y_tr = X[i_train], Y[i_train]
    X_va, Y_va = X[i_valid], Y[i_valid]
    X_te, Y_te = X[i_test],  Y[i_test]

    # Write out
    with h5py.File(out_path, "w") as fout:
        fout.create_dataset("idx_train", data=i_train, compression=compression)
        fout.create_dataset("idx_valid", data=i_valid, compression=compression)
        fout.create_dataset("idx_test",  data=i_test,  compression=compression)

        fout.create_dataset("x_train", data=X_tr, compression=compression, shuffle=True)
        fout.create_dataset("y_train", data=Y_tr, compression=compression, shuffle=True)
        fout.create_dataset("x_valid", data=X_va, compression=compression, shuffle=True)
        fout.create_dataset("y_valid", data=Y_va, compression=compression, shuffle=True)
        fout.create_dataset("x_test",  data=X_te, compression=compression, shuffle=True)
        fout.create_dataset("y_test",  data=Y_te, compression=compression, shuffle=True)

        fout.attrs["source_file"] = str(in_path)
        fout.attrs["ratios"] = ratios
        if seed is not None:
            fout.attrs["seed"] = seed

    return out_path


def split_h5_dataset(
    in_path,
    out_path=None,
    ratios=(0.8, 0.1, 0.1),
    seed=None,
    x_key="x",
    y_key="y",
    compression="gzip",
):
    """Reserve the first sample as wild type (x_wt/y_wt), split the rest by ratios, and write splits."""
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("ratios must sum to 1")

    in_path = Path(in_path)
    if out_path is None:
        out_path = in_path.with_name(in_path.stem + "_split.h5")
    out_path = Path(out_path)

    rng = np.random.default_rng(seed)

    # Load
    with h5py.File(in_path, "r") as fin:
        X = np.asarray(fin[x_key])  # (N, ...)
        Y = np.asarray(fin[y_key])  # (N, ...) or (N,)
    N = X.shape[0]
    if Y.shape[0] != N:
        raise ValueError("x and y must have the same first dimension")
    if N < 2:
        raise ValueError("need at least 2 samples to reserve the first as wild type")

    # Wild type (index 0)
    X_wt, Y_wt = X[0], Y[0]
    # Ensure non-scalar datasets so filters (compression/shuffle) are allowed
    X_wt_save = np.asarray(X_wt)
    if X_wt_save.ndim == 0:
        X_wt_save = X_wt_save[None]
    Y_wt_save = np.asarray(Y_wt)
    if Y_wt_save.ndim == 0:
        Y_wt_save = Y_wt_save[None]

    # Remaining indices [1..N-1]
    base_idx = np.arange(1, N)
    perm = rng.permutation(base_idx.size)
    perm_idx = base_idx[perm]

    n_train = int(ratios[0] * base_idx.size)
    n_valid = int(ratios[1] * base_idx.size)
    i_train = perm_idx[:n_train]
    i_valid = perm_idx[n_train:n_train + n_valid]
    i_test  = perm_idx[n_train + n_valid:]

    # Slice
    X_tr, Y_tr = X[i_train], Y[i_train]
    X_va, Y_va = X[i_valid], Y[i_valid]
    X_te, Y_te = X[i_test],  Y[i_test]

    # Write
    with h5py.File(out_path, "w") as fout:
        # Original indices per split (0 excluded)
        fout.create_dataset("idx_train", data=i_train, compression=compression)
        fout.create_dataset("idx_valid", data=i_valid, compression=compression)
        fout.create_dataset("idx_test",  data=i_test,  compression=compression)

        # Wild type
        fout.create_dataset("x_wt", data=X_wt_save, compression=compression, shuffle=True)
        fout.create_dataset("y_wt", data=Y_wt_save, compression=compression, shuffle=True)

        # Splits
        fout.create_dataset("x_train", data=X_tr, compression=compression, shuffle=True)
        fout.create_dataset("y_train", data=Y_tr, compression=compression, shuffle=True)
        fout.create_dataset("x_valid", data=X_va, compression=compression, shuffle=True)
        fout.create_dataset("y_valid", data=Y_va, compression=compression, shuffle=True)
        fout.create_dataset("x_test",  data=X_te, compression=compression, shuffle=True)
        fout.create_dataset("y_test",  data=Y_te, compression=compression, shuffle=True)

        fout.attrs["source_file"] = str(in_path)
        fout.attrs["ratios"] = ratios
        fout.attrs["wt_index"] = 0
        if seed is not None:
            fout.attrs["seed"] = seed

    return out_path
