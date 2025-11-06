import argparse
import h5py
import warnings
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


from insilico_mpra.generate import generate_mutagenesis_library, add_adapters
from insilico_mpra.predict import load_models
from insilico_mpra.utils.dna_utils import seq2onehot


torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate mutagenesis library")
    parser.add_argument(
        "--input", "-t", type=Path, required=True,
        help="Path to input file (tsv, csv, fasta) containing sequences to mutagenize.")
    parser.add_argument(
        "--format", "-f", type=str, choices=["tsv", "csv", "fasta"], default="tsv",
        help="Format of the input file (tsv, csv, fasta). Default is 'tsv'.")
    parser.add_argument(
        "--weights_dir", "-d", type=Path, required=True,
        help="Path to the weight directory where checkpoints are stored.")
    parser.add_argument(
        "--config_path", "-c", type=Path, default=None,
        help=("Path to the model configuration JSON file. "
              "If not provided, defaults to 'config.json' in weights_dir."))
    parser.add_argument(
        "--mut_rate", "-m", type=float, default=0.1,
        help="Mutation rate for the mutagenesis process. Default is 0.1.")
    parser.add_argument(
        "--num", "-n", type=int, default=1000,
        help="Number of sequences to generate. Default is 1000.")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1024,
        help="Batch size for prediction. Default is 1024.")
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
                continue  # skip blank lines
            if line[0] == ">":  # new header
                if hdr is not None:  # save previous record
                    records[hdr] = "".join(seq_parts)
                hdr = line[1:].split()[0]  # drop leading ">" + trim extras
                seq_parts = []
            else:
                seq_parts.append(line)
        if hdr is not None:  # flush last record
            records[hdr] = "".join(seq_parts)
    return records


def get_onehot_dict(input_path: Path | str, format: str = "fasta"):
    if format not in ["tsv", "csv", "fasta"]:
        raise ValueError(
            f"Unsupported data format: {format}. Supported formats are 'tsv', 'csv', and 'fasta'."
        )

    onehot_dict = {}
    if format in ["tsv", "csv"]:
        if format == "tsv":
            df = pd.read_csv(input_path, sep="\t")
        else:
            df = pd.read_csv(input_path)

        use_header = "header" in df.columns and df["header"].notna().all()

        for i in range(len(df)):
            seq = df.loc[i, "sequence"]
            if len(seq) != 200:
                warnings.warn(f"Sequence at index {i} is not of length 200. Skipping.", UserWarning)
                continue
            seq_with_adapters = add_adapters(seq)
            key = df.loc[i, "header"] if use_header else f"mpra{i}"
            onehot_dict[key] = seq2onehot(seq_with_adapters).transpose(1, 0)
    else:  # fasta format
        fasta_dict = read_fasta(input_path)
        for key, seq in fasta_dict.items():
            if len(seq) != 200:
                raise ValueError(f"Sequence {key} is not of length 200.")
            seq_with_adapters = add_adapters(seq)
            onehot_dict[key] = seq2onehot(seq_with_adapters).transpose(1, 0)

    return onehot_dict


def plot_y_hist(y_mut):
    """Function for visualizing histogram of inferred predictions for MAVE dataset.

    Parameters
    ----------
    y_mut : numpy.ndarray
        Inferred predictions for sequences (shape: (N,1)).

    Returns
    -------
    matplotlib.pyplot.Figure
    """
    # plot histogram of transformed deepnet predictions
    fig, ax = plt.subplots()
    ax.hist(y_mut, bins=100)
    ax.set_xlabel('y')
    ax.set_ylabel('Frequency')
    ax.axvline(y_mut[0], c='red', label='WT', linewidth=2, zorder=10) #wild-type prediction
    plt.legend(loc='upper right')
    plt.tight_layout()
    return fig


def main(
    infile: Path,
    format: str,
    weights_dir: Path,
    config_path: Path | None,
    outdir: Path,
    mut_rate: float,
    num: int,
    batch_size: int,
):
    print("===== Starting mutagenesis library generation =====")
    if not infile.exists():
        raise FileNotFoundError(f"Input TSV file {infile} does not exist.")

    weights_dir = weights_dir
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weight directory {weights_dir} does not exist.")

    outdir = outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Prepare the sequences
    print("====== Loading sequences from TSV file ======")
    onehot_dict = get_onehot_dict(infile, format=format)

    # Load models
    print("====== Loading models ======")
    if config_path is None:
        config_path = weights_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not provided and config.json not found in {weights_dir}."
            )
    models = load_models(config_path, weights_dir)

    print("====== Generating mutagenesis library ======")
    for key, onehot_seq in onehot_dict.items():
        print(f'----- Processing sequence "{key}" -----')
        x_mut, y_mut = generate_mutagenesis_library(
            x=onehot_seq, models=models, num=num, mut_rate=mut_rate, batch_size=batch_size
        )

        # chrom_start_end.h5
        outfile = outdir / f"{key}.h5"
        with h5py.File(outfile, "w") as f:
            f.create_dataset("x", data=x_mut)
            f.create_dataset("y", data=y_mut)
        print(f"Saved {outfile}")

        # Save figure
        fig = plot_y_hist(y_mut)
        fig.savefig(outfile.with_suffix(".png"), dpi=300)
        plt.close(fig)

    print("===== Mutagenesis library generation completed =====")


if __name__ == "__main__":
    args = parse_args()
    main(
        infile=args.input,
        format=args.format,
        weights_dir=args.weights_dir,
        config_path=args.config_path,
        outdir=args.outdir,
        mut_rate=args.mut_rate,
        num=args.num,
        batch_size=args.batch_size,
    )
