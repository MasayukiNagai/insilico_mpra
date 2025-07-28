# -*- coding: utf-8 -*-
"""
EvoAug Lightning Module for genomic data augmentation.

This module provides a PyTorch Lightning wrapper that integrates EvoAug's
evolution-inspired data augmentations with Lightning training workflows.
"""

import torch
import lightning.pytorch as pl
import numpy as np


# EvoAug imports (conditional to avoid hard dependency)
try:
    from evoaug import augment
    EVOAUG_AVAILABLE = True
except ImportError:
    EVOAUG_AVAILABLE = False


def create_evoaug_augment_list(config):
    """Create list of EvoAug augmentations based on configuration.
    
    Args:
        config: Configuration object with EvoAug parameters
        
    Returns:
        list: List of EvoAug augmentation objects
    """
    if not EVOAUG_AVAILABLE:
        return []
    
    augment_list = []
    
    # Map operation names to EvoAug augmentation classes
    operation_map = {
        'mutation': lambda: augment.RandomMutation(mutate_frac=getattr(config, 'mut_frac', 0.05)),
        'deletion': lambda: augment.RandomDeletion(
            delete_min=getattr(config, 'delete_min', 0), 
            delete_max=getattr(config, 'delete_max', 20)
        ),
        'insertion': lambda: augment.RandomInsertion(
            insert_min=getattr(config, 'insert_min', 0),
            insert_max=getattr(config, 'insert_max', 20)
        ),
        'translocation': lambda: augment.RandomTranslocation(
            shift_min=getattr(config, 'shift_min', 0),
            shift_max=getattr(config, 'shift_max', 20)
        ),
        'inversion': lambda: augment.RandomInversion(),
        'rc': lambda: augment.RandomRC(rc_prob=getattr(config, 'rc_prob', 0.5)),
        'noise': lambda: augment.RandomNoise(
            noise_mean=getattr(config, 'noise_mean', 0.0),
            noise_std=getattr(config, 'noise_std', 0.2)
        )
    }
    
    # Get operations list from config
    evoaug_ops = getattr(config, 'evoaug_ops', [])
    
    # Create augmentations based on specified operations
    for op_name in evoaug_ops:
        op_name = op_name.strip()
        if op_name in operation_map:
            aug = operation_map[op_name]()
            augment_list.append(aug)
            print(f"  Added EvoAug operation: {op_name}")
        else:
            print(f"  Warning: Unknown EvoAug operation '{op_name}', skipping")
    
    return augment_list


class EvoAugLitModel(pl.LightningModule):
    """PyTorch Lightning wrapper for EvoAug that ensures proper inheritance.
    
    This class provides a Lightning-compatible interface for training genomic models
    with EvoAug's evolution-inspired data augmentations. It replicates the core
    functionality of EvoAug's RobustModel while maintaining full Lightning compatibility.
    
    Args:
        lit_model: Original LightningModule to wrap
        config: Configuration object with training and EvoAug parameters
        augment_list: Optional list of augmentations (if None, will be created from config)
    """
    
    def __init__(self, lit_model, config, augment_list=None):
        super().__init__()
        self.config = config
        self.pytorch_model = lit_model.model
        self.criterion = lit_model.loss
        self.val_pearson = lit_model.val_pearson
        
        # Create or use provided augmentation list
        if augment_list is None:
            self.augment_list = create_evoaug_augment_list(config)
        else:
            self.augment_list = augment_list
        
        # EvoAug parameters
        self.max_augs_per_seq = min(
            getattr(config, 'max_augs_per_seq', 2), 
            len(self.augment_list)
        )
        self.hard_aug = getattr(config, 'hard_aug', True)
        self.inference_aug = getattr(config, 'inference_aug', False)
        self.finetune = False
        
        # EvoAug helper attributes
        self.max_num_aug = len(self.augment_list)
        self.insert_max = self._augment_max_len(self.augment_list)
        
        # Save hyperparameters
        self.save_hyperparameters({
            "config": config.to_dict() if hasattr(config, 'to_dict') else str(config),
            "num_augmentations": len(self.augment_list),
            "augmentation_types": [type(aug).__name__ for aug in self.augment_list]
        })
    
    def _augment_max_len(self, augment_list):
        """Determine insert_max from augmentation list.
        
        Args:
            augment_list: List of augmentation objects
            
        Returns:
            int: Maximum insertion length from augmentations
        """
        insert_max = 0
        for aug in augment_list:
            if hasattr(aug, 'insert_max'):
                insert_max = aug.insert_max
        return insert_max
    
    def _sample_aug_combos(self, batch_size):
        """Sample augmentation combinations for each sequence in batch.
        
        Args:
            batch_size: Number of sequences in batch
            
        Returns:
            list: List of augmentation index combinations for each sequence
        """
        if self.max_augs_per_seq == 0 or len(self.augment_list) == 0:
            return [[] for _ in range(batch_size)]
        
        # Determine number of augmentations per sequence
        if self.hard_aug:
            batch_num_aug = self.max_augs_per_seq * np.ones((batch_size,), dtype=int)
        else:
            batch_num_aug = np.random.randint(1, self.max_augs_per_seq + 1, (batch_size,))
        
        # Randomly choose augmentation combinations
        aug_combos = [
            list(sorted(np.random.choice(self.max_num_aug, sample, replace=False))) 
            for sample in batch_num_aug
        ]
        return aug_combos
    
    def _apply_augment(self, x):
        """Apply augmentations to each sequence in batch.
        
        Args:
            x: Input tensor of shape (batch_size, channels, length)
            
        Returns:
            torch.Tensor: Augmented sequences
        """
        if len(self.augment_list) == 0:
            return x
        
        aug_combos = self._sample_aug_combos(x.shape[0])
        
        x_new = []
        for aug_indices, seq in zip(aug_combos, x):
            seq = torch.unsqueeze(seq, dim=0)
            insert_status = True  # Track if padding is needed
            
            # Apply each augmentation in the combination
            for aug_index in aug_indices:
                seq = self.augment_list[aug_index](seq)
                if hasattr(self.augment_list[aug_index], 'insert_max'):
                    insert_status = False
            
            # Add padding if needed and no insertions were applied
            if insert_status and self.insert_max:
                seq = self._pad_end(seq)
            
            x_new.append(seq)
        
        return torch.cat(x_new)
    
    def _pad_end(self, x):
        """Add random DNA padding to end of sequences.
        
        Args:
            x: Input tensor of shape (batch_size, channels, length)
            
        Returns:
            torch.Tensor: Padded sequences
        """
        N, A, L = x.shape
        
        # Create random DNA padding
        a = torch.eye(A)
        p = torch.tensor([1/A for _ in range(A)])
        padding = torch.stack([
            a[p.multinomial(self.insert_max, replacement=True)].transpose(0, 1) 
            for _ in range(N)
        ]).to(x.device)
        
        x_padded = torch.cat([x, padding.to(x.device)], dim=2)
        return x_padded
    
    def forward(self, x):
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Model predictions
        """
        return self.pytorch_model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step with optional augmentations.
        
        Args:
            batch: Training batch (x, y)
            batch_idx: Batch index
            
        Returns:
            torch.Tensor: Training loss
        """
        x, y = batch
        
        # Apply augmentations unless in fine-tuning mode
        if self.finetune:
            if self.insert_max:
                x = self._pad_end(x)
        else:
            if len(self.augment_list) > 0:
                x = self._apply_augment(x)
        
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with optional augmentations.
        
        Args:
            batch: Validation batch (x, y)
            batch_idx: Batch index
            
        Returns:
            torch.Tensor: Validation loss
        """
        x, y = batch
        
        # Apply augmentations if inference_aug is enabled
        if self.inference_aug:
            if len(self.augment_list) > 0:
                x = self._apply_augment(x)
        else:
            if self.insert_max:
                x = self._pad_end(x)
        
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.val_pearson(y_hat, y)
        self.log("val_pearson", self.val_pearson, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step with optional augmentations.
        
        Args:
            batch: Test batch (x, y)
            batch_idx: Batch index
            
        Returns:
            dict: Test results
        """
        x, y = batch
        
        # Apply augmentations if inference_aug is enabled
        if self.inference_aug:
            if len(self.augment_list) > 0:
                x = self._apply_augment(x)
        else:
            if self.insert_max:
                x = self._pad_end(x)
        
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return {'test_loss': loss, 'predictions': y_hat, 'targets': y}
    
    def predict_step(self, batch, batch_idx):
        """Prediction step with optional augmentations.
        
        Args:
            batch: Input batch (x,) or (x, y)
            batch_idx: Batch index
            
        Returns:
            torch.Tensor: Model predictions
        """
        if isinstance(batch, (tuple, list)):
            x, _ = batch
        else:
            x = batch
        
        # Apply augmentations if inference_aug is enabled
        if self.inference_aug:
            if len(self.augment_list) > 0:
                x = self._apply_augment(x)
        else:
            if self.insert_max:
                x = self._pad_end(x)
        
        return self(x)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            tuple: (optimizers, schedulers)
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=getattr(self.config, 'max_lr', 1e-3) / 25, # / 25,
            weight_decay=getattr(self.config, 'weight_decay', 1e-6)
        )
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=getattr(self.config, 'max_lr', 1e-3),
            three_phase=False,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            cycle_momentum=False
        )
        
        # Try ReduceLROnPlateau
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='max',
        #     factor=0.3,
        #     patience=2,
        #     verbose=True
        # )
        
        # Try CosineAnnealingLR
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=getattr(self.config, 't_max', 10), # Number of epochs for cosine annealing
        #     eta_min=getattr(self.config, 'eta_min', 1e-6) # Minimum learning rate
        # )
        
        return [optimizer], [{
            "scheduler": lr_scheduler,
            # "monitor": "val_pearson",
            "interval": "step", # step
            "frequency": 1,
            "name": "cycle_lr" # cycle_lr
        }]
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('lr', current_lr, on_epoch=True)
    
    def finetune_mode(self):
        """Enable fine-tuning mode (disables augmentations during training)."""
        self.finetune = True
        print("Fine-tuning mode enabled - augmentations disabled for training")
    
    def inference_mode(self, use_augmentations=False):
        """Set inference mode for augmentations.
        
        Args:
            use_augmentations: Whether to use augmentations during inference
        """
        self.inference_aug = use_augmentations
        print(f"Inference augmentations {'enabled' if use_augmentations else 'disabled'}")
    
    def get_augmentation_info(self):
        """Get information about configured augmentations.
        
        Returns:
            dict: Augmentation configuration details
        """
        return {
            'num_augmentations': len(self.augment_list),
            'augmentation_types': [type(aug).__name__ for aug in self.augment_list],
            'max_augs_per_seq': self.max_augs_per_seq,
            'hard_aug': self.hard_aug,
            'inference_aug': self.inference_aug,
            'finetune_mode': self.finetune,
            'insert_max': self.insert_max
        }


def create_evoaug_model(lit_model, config):
    """Factory function to create EvoAug model wrapper.
    
    Args:
        lit_model: Original LightningModule to wrap
        config: Configuration object with EvoAug parameters
        
    Returns:
        EvoAugLitModel or original model: Wrapped model if EvoAug is enabled
    """
    use_evoaug = getattr(config, 'use_evoaug', False)
    
    if not use_evoaug:
        print("EvoAug disabled, using original model")
        return lit_model
    
    if not EVOAUG_AVAILABLE:
        print("Warning: EvoAug not available, using original model")
        return lit_model
    
    print("Creating EvoAug Lightning Model:")
    
    # Create augmentation list
    augment_list = create_evoaug_augment_list(config)
    
    if not augment_list:
        print("  No valid EvoAug operations specified, using original model")
        return lit_model
    
    # Create EvoAug wrapper
    evoaug_model = EvoAugLitModel(lit_model, config, augment_list)
    
    print(f"  Created EvoAug Lightning Model with {len(augment_list)} augmentations")
    print(f"  Max augs per sequence: {evoaug_model.max_augs_per_seq}")
    print(f"  Hard augmentation: {evoaug_model.hard_aug}")
    print(f"  Inference augmentation: {evoaug_model.inference_aug}")
    
    return evoaug_model