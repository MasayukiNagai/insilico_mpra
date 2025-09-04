import torch
import numpy as np
import pandas as pd
# import lightning.pytorch as pl
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from insilico_mpra.config import TrainingConfig
from insilico_mpra.models.lightning_module import LitModel
from insilico_mpra.data.datasets import HDF5Dataset
from insilico_mpra.data.utils import create_test_dataloader
from insilico_mpra.utils.dna_utils import Seq2Tensor


class SequenceDataset(Dataset):
    """Simple dataset for raw DNA sequences."""

    def __init__(self, sequences, config):
        """
        Initialize sequence dataset.

        Args:
            sequences: List of DNA sequence strings
            config: TrainingConfig object
        """
        self.sequences = sequences
        self.config = config
        self.totensor = Seq2Tensor()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Convert to tensor
        seq_tensor = self.totensor(seq)

        # Add reverse channel if needed
        if self.config.use_reverse_channel:
            rev_channel = torch.zeros(1, seq_tensor.shape[1])
            seq_tensor = torch.cat([seq_tensor, rev_channel], dim=0)

        return seq_tensor


class OneHotDataset(Dataset):
    """Dataset for one-hot encoded sequences."""

    def __init__(self, onehot):
        """
        Initialize dataset with one-hot encoded sequences and targets.

        Args:
            onehot: Numpy array of shape (n_samples, 4, seq_length)
        """
        self.onehot = torch.from_numpy(onehot).float()

    def __len__(self):
        return len(self.onehot)

    def __getitem__(self, idx):
        return self.onehot[idx]


def load_model(config_path, checkpoint_path):
    """Load trained model from checkpoint."""
    config = TrainingConfig.from_json(config_path, training=False)
    model = LitModel(tr_cfg=config)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['state_dict'])
    return model, config


def load_sequences_from_file(file_path):
    """Load sequences from text file."""
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            seq = line.strip()
            if seq:  # Skip empty lines
                sequences.append(seq)
    return sequences


def load_data_from_hdf5(file_path, split='test', use_reverse_channel=False):
    """Load sequences from HDF5 file."""
    ds = HDF5Dataset(
        file_path,
        split=split,
        use_reverse_channel=use_reverse_channel,
        training=False
    )
    return ds.sequences, ds.targets


def predict_from_hdf5(model, config, data_file, split='test', batch_size=1024, num_workers=4):
    """Make predictions on HDF5 dataset."""
    # Create test dataloader
    test_dl = create_test_dataloader(config, data_file, split=split)

    # Setup trainer for prediction
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[config.device] if torch.cuda.is_available() else 1,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True
    )

    # Get predictions
    predictions = trainer.predict(model, test_dl)
    y_pred = torch.cat(predictions).cpu().numpy()

    # Get true values for comparison
    dataset = test_dl.dataset
    y_true = dataset.targets

    return y_pred, y_true


def predict_from_sequences(model, config, sequences, batch_size=1024, num_workers=4):
    """Make predictions on raw sequences."""
    # Create dataset
    dataset = SequenceDataset(sequences)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Setup trainer for prediction
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[config.device] if torch.cuda.is_available() else 1,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True
    )

    # Get predictions
    predictions = trainer.predict(model, dataloader)
    y_pred = torch.cat(predictions).cpu().numpy()

    return y_pred


def predict_from_onehot(model, onehot, batch_size=1024, num_workers=4):
    """Make predictions on raw sequences."""
    # Create dataset
    dataset = OneHotDataset(onehot)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Setup trainer for prediction
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto', # 'cuda' if torch.cuda.is_available() else 1,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    # Get predictions
    predictions = trainer.predict(model, dataloader)
    y_pred = torch.cat(predictions).cpu().numpy()

    return y_pred


def predict_with_reverse_complement(model, onehot, batch_size=1024, num_workers=4):
    """Make predictions with test-time augmentation."""
    from insilico_mpra.utils.dna_utils import reverse_complement_array

    # Forward predictions
    forward_preds = predict_from_onehot(model, onehot, batch_size, num_workers)

    rev_sequences = np.array([reverse_complement_array(array) for array in onehot])
    rev_preds = predict_from_onehot(model, rev_sequences, batch_size, num_workers)

    # Average forward and reverse predictions
    return (forward_preds + rev_preds) / 2


def save_predictions(predictions, output_path, sequences=None, targets=None):
    """Save predictions to file."""
    data = {'predictions': predictions}

    if sequences is not None:
        data['sequences'] = sequences

    if targets is not None:
        data['targets'] = targets

        # Calculate metrics if targets are available
        from scipy.stats import pearsonr
        from sklearn.metrics import mean_squared_error, r2_score

        pearson_corr, _ = pearsonr(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        print(f"\nPrediction Metrics:")
        print(f"Pearson Correlation: {pearson_corr:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")

    # Save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def predict_ensemble_from_onehot(models, onehot):
    """
    Parameters
    ----------
    models: list[pl.LightningModule]   pre-loaded models
    onehot: np.ndarray | torch.Tensor  shape (N, C, L)

    Returns
    -------
    np.ndarray   ensemble-averaged predictions for the input
    """
    # 1 -- device & mixed-precision context
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2 -- prepare input  (add batch-dim, move to device)
    x = torch.as_tensor(onehot, dtype=torch.float32, device=device)

    # 3 -- run every model, collect outputs
    preds = []
    with torch.no_grad():
        for model in models:
            model = model.to(device).eval()
            if hasattr(model, "predict_step"):
                y = model.predict_step(x, 0)
            else:
                y = model(x)
            preds.append(y.detach().cpu())

    # 4 -- average and return numpy
    return torch.mean(torch.stack(preds), dim=0).numpy()
