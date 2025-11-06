# -*- coding: utf-8 -*-
"""
PyTorch Lightning module for MPRA LegNet training and validation.
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl
# import pytorch_lightning as pl
from torchmetrics import PearsonCorrCoef
from ..utils.model_utils import initialize_weights


class LitModel(pl.LightningModule):
    """PyTorch Lightning module for LegNet training."""

    def __init__(self, tr_cfg):
        """
        Initialize Lightning module.

        Args:
            tr_cfg: TrainingConfig object with model and training parameters
        """
        super().__init__()
        self.tr_cfg = tr_cfg
        self.model = self.tr_cfg.get_model()
        self.model.apply(initialize_weights)
        self.loss = nn.MSELoss()
        self.val_pearson = PearsonCorrCoef()

        # Save hyperparameters
        self.save_hyperparameters({"config": self.tr_cfg.to_dict()})

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = batch
        y_hat = self.model(X)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.val_pearson(y_hat, y)
        self.log("val_pearson", self.val_pearson, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return {'test_loss': loss, 'predictions': y_hat, 'targets': y}

    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        if isinstance(batch, (tuple, list)):
            x, _ = batch
        else:
            x = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.tr_cfg.max_lr / 25,
            weight_decay=self.tr_cfg.weight_decay
        )

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.tr_cfg.max_lr,
            three_phase=False,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            cycle_momentum=False
        )

        return [optimizer], [{
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "cycle_lr"
        }]

    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Log learning rate
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('lr', current_lr, on_epoch=True)
