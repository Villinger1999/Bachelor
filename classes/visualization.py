"""
Visualization module for flexible visualization of attacks, reconstructions, and training.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Dict
from classes.helperfunctions import fix_dimension


class Visualizer:
    """Flexible visualization class for various scenarios."""
    
    @staticmethod
    def visualize_reconstruction(
        original: Union[torch.Tensor, np.ndarray],
        reconstructed: Union[torch.Tensor, np.ndarray],
        dummy: Optional[Union[torch.Tensor, np.ndarray]] = None,
        losses: Optional[List[float]] = None,
        history: Optional[List] = None,
        pred_label: Optional[torch.Tensor] = None,
        true_label: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None,
        save_path: Optional[str] = None,
        show: bool = False,
        figsize: tuple = (20, 5)
    ):
        """
        Visualize image reconstruction attack results.
        
        Args:
            original: Original image
            reconstructed: Reconstructed image
            dummy: Initial dummy image (optional)
            losses: Loss history (optional)
            history: Reconstruction history (optional)
            pred_label: Predicted label (optional)
            true_label: True label (optional)
            metadata: Additional metadata dict for title
            save_path: Path to save figure
            show: Whether to display figure
            figsize: Figure size
        """
        # Convert to displayable format
        orig_display = fix_dimension(original)
        recon_display = fix_dimension(reconstructed)
        
        # Determine number of subplots
        num_plots = 2  # original + reconstructed
        if dummy is not None:
            num_plots += 1
        if losses is not None:
            num_plots += 1
        
        fig, axes = plt.subplots(1, num_plots, figsize=figsize)
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Dummy initialization
        if dummy is not None:
            dummy_display = fix_dimension(dummy)
            axes[plot_idx].imshow(dummy_display)
            axes[plot_idx].set_title("Dummy Initialization")
            axes[plot_idx].axis("off")
            plot_idx += 1
        
        # Reconstructed image
        axes[plot_idx].imshow(recon_display)
        title = "Reconstructed"
        if pred_label is not None:
            title += f"\n(Pred: {pred_label.item()})"
        axes[plot_idx].set_title(title)
        axes[plot_idx].axis("off")
        plot_idx += 1
        
        # Original image
        axes[plot_idx].imshow(orig_display)
        title = "Original"
        if true_label is not None:
            title += f"\n(Label: {true_label.item()})"
        axes[plot_idx].set_title(title)
        axes[plot_idx].axis("off")
        plot_idx += 1
        
        # Loss curve
        if losses is not None:
            axes[plot_idx].plot(losses)
            axes[plot_idx].set_title("Loss Curve")
            axes[plot_idx].set_xlabel("Iteration")
            axes[plot_idx].set_ylabel("Loss")
            axes[plot_idx].grid(True)
        
        # Add metadata to suptitle if provided
        if metadata:
            metadata_str = " | ".join([f"{k}: {v}" for k, v in metadata.items()])
            fig.suptitle(metadata_str, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    @staticmethod
    def visualize_training_history(
        train_losses: Optional[List[float]] = None,
        train_accuracies: Optional[List[float]] = None,
        val_losses: Optional[List[float]] = None,
        val_accuracies: Optional[List[float]] = None,
        save_path: Optional[str] = None,
        show: bool = False,
        figsize: tuple = (12, 5)
    ):
        """
        Visualize training history (loss and accuracy curves).
        
        Args:
            train_losses: Training loss history
            train_accuracies: Training accuracy history
            val_losses: Validation loss history
            val_accuracies: Validation accuracy history
            save_path: Path to save figure
            show: Whether to display figure
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss plot
        if train_losses is not None:
            axes[0].plot(train_losses, label='Train', marker='o')
        if val_losses is not None:
            axes[0].plot(val_losses, label='Val', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        if train_accuracies is not None:
            axes[1].plot(train_accuracies, label='Train', marker='o')
        if val_accuracies is not None:
            axes[1].plot(val_accuracies, label='Val', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    @staticmethod
    def visualize_gradient_comparison(
        original_grads: List[torch.Tensor],
        modified_grads: List[torch.Tensor],
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Visualize comparison between original and modified gradients.
        
        Args:
            original_grads: List of original gradient tensors
            modified_grads: List of modified gradient tensors
            save_path: Path to save figure
            show: Whether to display figure
        """
        # Flatten all gradients
        orig_flat = torch.cat([g.detach().abs().flatten() for g in original_grads])
        mod_flat = torch.cat([g.detach().abs().flatten() for g in modified_grads])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].hist(orig_flat.cpu().numpy(), bins=50, alpha=0.7, label='Original')
        axes[0].hist(mod_flat.cpu().numpy(), bins=50, alpha=0.7, label='Modified')
        axes[0].set_xlabel('Gradient Magnitude')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Gradient Distribution')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].scatter(orig_flat.cpu().numpy(), mod_flat.cpu().numpy(), alpha=0.5)
        axes[1].plot([orig_flat.min(), orig_flat.max()], 
                     [orig_flat.min(), orig_flat.max()], 'r--', label='y=x')
        axes[1].set_xlabel('Original Gradient Magnitude')
        axes[1].set_ylabel('Modified Gradient Magnitude')
        axes[1].set_title('Gradient Comparison')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)

