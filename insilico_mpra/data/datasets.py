# -*- coding: utf-8 -*-
"""
Dataset classes for MPRA data in different formats.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset
from ..utils.dna_utils import Seq2Tensor, reverse_complement


class HDF5Dataset(Dataset):
    """Dataset for HDF5 format MPRA data."""

    def __init__(self, h5_file, split='train', use_reverse=False, use_shift=False,
                 use_reverse_channel=False, max_shift=None, seqsize=230, training=True):
        """
        Initialize HDF5 dataset.
        
        Args:
            h5_file: Path to HDF5 file
            split: Data split ('train', 'valid', 'test')
            use_reverse: Whether to apply reverse complement augmentation
            use_shift: Whether to apply shift augmentation
            use_reverse_channel: Whether to add reverse complement indicator channel
            max_shift: Maximum shift range (tuple)
            seqsize: Expected sequence size
            training: Whether in training mode (affects augmentation)
        """
        self.h5_file = h5_file
        self.split = split
        self.use_reverse = use_reverse
        self.use_shift = use_shift
        self.use_reverse_channel = use_reverse_channel
        self.seqsize = seqsize
        self.training = training

        # Load data from HDF5
        with h5py.File(h5_file, 'r') as f:
            # Load one-hot encoded sequences
            self.sequences = f[f'onehot_{split}'][:]  # shape: (n, 230, 4)
            self.targets = f[f'y_{split}'][:].squeeze()  # shape: (n,)

        # Convert to proper format: (n, 4, 230)
        self.sequences = np.transpose(self.sequences, (0, 2, 1))

        # Shift parameters
        self.forward_side = "GGCCCGCTCTAGACCTGCAGG"
        self.reverse_side = "CACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGT"
        if max_shift is None:
            self.max_shift = (0, len(self.forward_side))
        else:
            self.max_shift = max_shift

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Get a single sample."""
        # Get sequence as tensor (4, 230)
        seq = torch.from_numpy(self.sequences[idx]).float()
        target = self.targets[idx]

        # Apply augmentations during training
        if self.training:
            # Shift augmentation (simplified for one-hot data)
            if self.use_shift and self.max_shift[1] > 0:
                shift = torch.randint(low=-self.max_shift[0], high=self.max_shift[1] + 1, size=(1,)).item()
                if shift != 0:
                    # Simple circular shift for demonstration
                    seq = torch.roll(seq, shift, dims=-1)

            # Reverse complement augmentation
            if self.use_reverse:
                if torch.rand(1).item() > 0.5:
                    # Reverse complement: flip both sequence and nucleotide order
                    seq = torch.flip(seq, dims=(-1, -2))
                    rev = 1.0
                else:
                    rev = 0.0
            else:
                rev = 0.0
        else:
            rev = 0.0

        # Add reverse channel if needed
        to_concat = [seq]
        if self.use_reverse_channel:
            rev_channel = torch.full((1, self.seqsize), rev, dtype=torch.float32)
            to_concat.append(rev_channel)

        # Create final tensor
        if len(to_concat) > 1:
            X = torch.concat(to_concat, dim=0)
        else:
            X = seq

        return X, target.astype(np.float32)


class TSVDataset(Dataset):
    """Dataset for TSV format data (legacy support)."""

    def __init__(self, df, use_reverse=False, use_shift=False, use_reverse_channel=False,
                 max_shift=None, seqsize=230, training=True):
        """
        Initialize TSV dataset.
        
        Args:
            df: Pandas DataFrame with sequence data
            use_reverse: Whether to apply reverse complement augmentation
            use_shift: Whether to apply shift augmentation
            use_reverse_channel: Whether to add reverse complement indicator channel
            max_shift: Maximum shift range (tuple)
            seqsize: Expected sequence size
            training: Whether in training mode (affects augmentation)
        """
        self.df = df
        self.totensor = Seq2Tensor()
        self.use_reverse = use_reverse
        self.use_shift = use_shift
        self.use_reverse_channel = use_reverse_channel
        self.seqsize = seqsize
        self.training = training

        self.forward_side = "GGCCCGCTCTAGACCTGCAGG"
        self.reverse_side = "CACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGT"

        if max_shift is None:
            self.max_shift = (0, len(self.forward_side))
        else:
            self.max_shift = max_shift

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get a single sample."""
        seq = self.df.seq.iloc[idx]
        target = self.df.mean_value.iloc[idx]

        # Apply shift augmentation
        if self.training and self.use_shift:
            shift = torch.randint(low=-self.max_shift[0], high=self.max_shift[1] + 1, size=(1,)).item()
            if shift < 0:
                seq = seq[:shift]
                seq = self.forward_side[shift:] + seq
            elif shift > 0:
                seq = seq[shift:]
                seq = seq + self.reverse_side[:shift]

        # Apply reverse complement
        if self.training and self.use_reverse:
            if torch.rand(1).item() > 0.5:
                seq = reverse_complement(seq)
                rev = 1.0
            else:
                rev = 0.0
        else:
            rev = 0.0

        # Convert to tensor
        seq = self.totensor(seq)
        to_concat = [seq]

        # Add reverse channel
        if self.use_reverse_channel:
            rev_channel = torch.full((1, self.seqsize), rev, dtype=torch.float32)
            to_concat.append(rev_channel)

        if len(to_concat) > 1:
            X = torch.concat(to_concat, dim=0)
        else:
            X = seq

        return X, target.astype(np.float32) 