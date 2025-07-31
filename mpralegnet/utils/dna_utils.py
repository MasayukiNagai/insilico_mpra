# -*- coding: utf-8 -*-
"""
DNA sequence utilities for encoding and manipulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# DNA encoding constants
CODES = {"A": 0, "T": 3, "G": 1, "C": 2, 'N': 4}
INV_CODES = {value: key for key, value in CODES.items()}
COMPL = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}


def n2id(n):
    """Convert nucleotide to integer ID."""
    return CODES[n.upper()]


def n2compl(n):
    """Get complement of nucleotide."""
    return COMPL[n.upper()]


def reverse_complement(seq, mapping={"A": "T", "G": "C", "T": "A", "C": "G", 'N': 'N'}):
    """Get reverse complement of DNA sequence."""
    return "".join(mapping[s] for s in reversed(seq))


def reverse_complement_array(array):
    """
    Get reverse complement of one-hot DNA encoding.

    Args:
        encoded_seq: np.ndarray of shape
            - (4, L), or
            - (N, 4, L),
          where the channel axis is ordered [A, C, G, T].

    Returns:
        np.ndarray of the same shape, reverse-complemented.
    """
    if array.ndim == 2:
        # single sequence
        return array[:, ::-1][[3, 2, 1, 0], :]

    if array.ndim == 3:
        # batch of sequences
        return array[:, :, ::-1][:, [3, 2, 1, 0], :]

    raise ValueError(f"Expected 2D or 3D array, got shape {array.shape}")


class Seq2Tensor(nn.Module):
    """Convert DNA sequence to one-hot encoded tensor."""

    def __init__(self):
        super().__init__()

    def forward(self, seq):
        """
        Convert DNA sequence to tensor.

        Args:
            seq: DNA sequence string or existing tensor

        Returns:
            Tensor of shape (4, sequence_length) with one-hot encoding
        """
        if isinstance(seq, torch.FloatTensor):
            return seq

        seq = [n2id(x) for x in seq]
        code = torch.from_numpy(np.array(seq))
        code = F.one_hot(code, num_classes=5)
        code[code[:, 4] == 1] = 0.25  # encode Ns with .25
        code = code[:, :4].float()
        return code.transpose(0, 1)
