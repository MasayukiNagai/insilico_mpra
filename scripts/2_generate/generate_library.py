import argparse
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from squid.mutagenizer import RandomMutagenesis
from squid.predictor import ScalarPredictor
from squid.mave import InSilicoMAVE
from squid.impress import plot_y_hist

from insilico_mpra.predict import load_model, predict_ensemble_from_onehot
from insilico_mpra.utils.dna_utils import seq2onehot

torch.set_float32_matmul_precision('medium')


def parse_args():
    parser = argparse.ArgumentParser(description="Train model")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--input_tsv", "-t", type=Path,
        help="Input TSV file containing sequences to mutagenize.")
    g.add_argument(
        "--input_fasta", "-f", type=Path,
        help="Input FASTA file containing sequences to mutagenize.")
    parser.add_argument(
        "--weights_dir", "-d", type=Path, required=True,
        help="Path to the weight directory where checkpoints are stored.")
    parser.add_argument(
        "--mut_rate", "-m", type=float, default=0.1,
        help="Mutation rate for the mutagenesis process.")
    parser.add_argument(
        "--num", "-n", type=int, default=1000,
        help="Number of sequences to generate.")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1024,
        help="Batch size for prediction.")
    parser.add_argument(
        "--num_workers", "-w", type=int, default=4,
        help="Number of workers for data loading.")
    parser.add_argument(
        "--outdir", "-o", type=Path, default=Path("output"),
        help="Output directory to save the generated mutagenesis library.")
    return parser.parse_args()


def read_fasta(path):
    records = {}
    with open(path, "r") as fh:
        hdr, seq_parts = None, []
        for line in fh:
            line = line.rstrip()
            if not line:
                continue                         # skip blank lines
            if line[0] == ">":                   # new header
                if hdr is not None:              # save previous record
                    records[hdr] = "".join(seq_parts)
                hdr = line[1:].split()[0]        # drop leading ">" + trim extras
                seq_parts = []
            else:
                seq_parts.append(line)
        if hdr is not None:                      # flush last record
            records[hdr] = "".join(seq_parts)
    return records


def load_models(weights_dir, num_models=10):
    models = []
    for i in range(num_models):
        model_path = Path(weights_dir) / str(i) / 'best.ckpt'
        config_path = Path(weights_dir) / str(i) / 'config.json'
        if model_path.exists() and config_path.exists():
            model, _ = load_model(config_path, model_path)
            models.append(model)
        else:
            raise FileNotFoundError(
                f"Model or config file not found for model {i} at {model_path} or {config_path}")
    return models


def add_adapters(sequence):
    adapter5 = 'AGGACCGGATCAACT'
    adapter3 = 'CATTGCGTGAACCGA'
    return adapter5 + sequence + adapter3


def generate_mutagenesis_library(
    x: np.ndarray,
    models,
    num=10_000,
    mut_rate=0.1,
    batch_size=1024,
    mut_window=[15, 215]
):
    """
    Generate a mutagenesis library using the provided models and mutation rate.

    Parameters
    ----------
    x : np.ndarray (L, 4)
        Input sequence in one-hot encoded format.
    models : list of torch.nn.Module
        List of pre-trained models to use for predictions.
    num : int
        Number of sequences to generate.
    mut_rate : float
        Mutation rate for the mutagenesis process.
    batch_size : int
        Batch size for processing the input sequences.
    """
    mut_generator = RandomMutagenesis(mut_rate=mut_rate)

    ensemble_fun = lambda x: predict_ensemble_from_onehot(
        models=models,
        onehot=x.transpose(0, 2, 1),
    )
    mut_predictor = ScalarPredictor(
        pred_fun=ensemble_fun,
        batch_size=batch_size
    )

    mave = InSilicoMAVE(
        mut_generator=mut_generator,
        mut_predictor=mut_predictor,
        seq_length=x.shape[0],
        mut_window=mut_window,
    )

    x_mut, y_mut = mave.generate(x, num_sim=num)

    return x_mut, y_mut


def main_tsv():
    print("Starting mutagenesis library generation...")
    args = parse_args()

    intsv = args.input_tsv
    if not intsv.exists():
        raise FileNotFoundError(f"Input TSV file {intsv} does not exist.")

    weights_dir = args.weights_dir
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weight directory {weights_dir} does not exist.")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(intsv, sep='\t')

    # Load models
    print("Loading models...")
    models = load_models(weights_dir, num_models=10)

    # Prepare the sequence
    onehot_dict = {}
    for i in df.index:
        seq = df.loc[i, 'sequence']
        assert len(seq) == 200, f"Sequence at index {i} is not of length 200."
        seq_with_adapters = add_adapters(seq)
        onehot_dict[i] = seq2onehot(seq_with_adapters).transpose(1, 0)

    # Generate mutagenesis library
    print("Generating mutagenesis library...")
    for i, onehot_seq in onehot_dict.items():
        print(f"Processing sequence {i}...")
        x_mut, y_mut = generate_mutagenesis_library(
            x=onehot_seq,
            models=models,
            num=args.num,
            mut_rate=args.mut_rate,
            batch_size=args.batch_size
        )

        # chrom_start_end.h5
        chrom = df.loc[i, 'chrom']
        start = df.loc[i, 'start']
        end = df.loc[i, 'end']
        outfile = outdir / f'{chrom}_{start}_{end}.h5'
        with h5py.File(outfile, 'w') as f:
            f.create_dataset('x', data=x_mut)
            f.create_dataset('y', data=y_mut)

        # Save figure
        fig = plot_y_hist(y_mut)
        fig.savefig(outfile.with_suffix('.png'), dpi=300)
        plt.close(fig)

def main_fasta():
    print("Starting mutagenesis library generation...")
    args = parse_args()

    infasta = args.input_fasta
    if not infasta.exists():
        raise FileNotFoundError(f"Input fasta file {infasta} does not exist.")

    weights_dir = args.weights_dir
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weight directory {weights_dir} does not exist.")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Load sequences from fasta file
    print("Loading sequences from fasta file...")
    fasta_dict = read_fasta(infasta)

    # Load models
    print("Loading models...")
    models = load_models(weights_dir, num_models=10)

    # Prepare the sequence
    onehot_dict = {}
    for k, seq in fasta_dict.items():
        assert len(seq) == 200, f"Sequence {k} is not of length 200."
        seq_with_adapters = add_adapters(seq)
        onehot_dict[k] = seq2onehot(seq_with_adapters).transpose(1, 0)  # (L, 4)

    # Generate mutagenesis library
    print("Generating mutagenesis library...")
    for k, onehot_seq in onehot_dict.items():
        print(f"Processing sequence {k}...")
        x_mut, y_mut = generate_mutagenesis_library(
            x=onehot_seq,
            models=models,
            num=args.num,
            mut_rate=args.mut_rate,
            batch_size=args.batch_size
        )

        output_file = outdir / f"{k}.h5"
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('x', data=x_mut)
            f.create_dataset('y', data=y_mut)

        # Save figure
        fig = plot_y_hist(y_mut)
        fig.savefig(output_file.with_suffix('.png'), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    if args.input_tsv:
        main_tsv()
    elif args.input_fasta:
        main_fasta()
