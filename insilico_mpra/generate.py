import numpy as np

from squid.mutagenizer import RandomMutagenesis
from squid.predictor import ScalarPredictor
from squid.mave import InSilicoMAVE

from insilico_mpra.predict import predict_ensemble_from_onehot


def add_adapters(sequence):
    adapter5 = "AGGACCGGATCAACT"
    adapter3 = "CATTGCGTGAACCGA"
    return adapter5 + sequence + adapter3


def generate_mutagenesis_library(
    x: np.ndarray, models, num=10_000, mut_rate=0.1, batch_size=1024, mut_window=[15, 215]
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
    mut_predictor = ScalarPredictor(pred_fun=ensemble_fun, batch_size=batch_size)

    mave = InSilicoMAVE(
        mut_generator=mut_generator,
        mut_predictor=mut_predictor,
        seq_length=x.shape[0],
        mut_window=mut_window,
    )

    x_mut, y_mut = mave.generate(x, num_sim=num)

    return x_mut, y_mut
