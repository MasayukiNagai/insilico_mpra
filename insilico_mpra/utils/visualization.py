# -*- coding: utf-8 -*-
"""
Visualization utilities for model evaluation and results plotting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr


def plot_predictions(y_true, y_pred, title="Predictions vs True Values", save_path=None):
    """
    Plot predictions vs true values with correlation information.
    
    Args:
        y_true: True values
        y_pred: Predicted values  
        title: Plot title
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)

    # Calculate and display correlation
    corr, _ = pearsonr(y_true, y_pred)
    plt.text(0.05, 0.95, f'Pearson r = {corr:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(train_losses, val_losses, val_pearson=None, save_path=None):
    """
    Plot training history including losses and validation metrics.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_pearson: Optional list of validation Pearson correlations per epoch
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(1, 2 if val_pearson is None else 3, figsize=(15, 5))
    
    # Plot losses
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot learning rate if available
    if len(axes) > 2:
        axes[1].plot(val_pearson, label='Validation Pearson', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Pearson Correlation')
        axes[1].set_title('Validation Performance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_sequence_importance(sequence, importance_scores, title="Sequence Importance", save_path=None):
    """
    Plot sequence importance scores as a heatmap.
    
    Args:
        sequence: DNA sequence string
        importance_scores: Importance scores for each position
        title: Plot title
        save_path: Optional path to save plot
    """
    fig, ax = plt.subplots(figsize=(min(len(sequence) * 0.1, 20), 3))
    
    # Create heatmap
    im = ax.imshow(importance_scores.reshape(1, -1), cmap='RdBu_r', aspect='auto')
    
    # Set labels
    ax.set_xticks(range(len(sequence)))
    ax.set_xticklabels(list(sequence))
    ax.set_yticks([])
    ax.set_title(title)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 