"""
Evaluation module for computing various metrics.
Supports PSNR, SSIM, loss, accuracy, and other metrics.
"""
import torch
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from typing import Dict, Optional, Union, List
from classes.helperfunctions import fix_dimension


class MetricsCalculator:
    """Calculate various evaluation metrics."""
    
    @staticmethod
    def calculate_psnr(
        original: Union[torch.Tensor, np.ndarray],
        reconstructed: Union[torch.Tensor, np.ndarray],
        data_range: float = 1.0
    ) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).
        
        Args:
            original: Original image tensor/array
            reconstructed: Reconstructed image tensor/array
            data_range: Data range (1.0 for [0,1], 255.0 for [0,255])
        
        Returns:
            PSNR value
        """
        orig = fix_dimension(original)
        recon = fix_dimension(reconstructed)
        return float(peak_signal_noise_ratio(orig, recon, data_range=data_range))
    
    @staticmethod
    def calculate_ssim(
        original: Union[torch.Tensor, np.ndarray],
        reconstructed: Union[torch.Tensor, np.ndarray],
        data_range: float = 1.0
    ) -> float:
        """
        Calculate Structural Similarity Index (SSIM).
        
        Args:
            original: Original image tensor/array
            reconstructed: Reconstructed image tensor/array
            data_range: Data range (1.0 for [0,1], 255.0 for [0,255])
        
        Returns:
            SSIM value
        """
        orig = fix_dimension(original)
        recon = fix_dimension(reconstructed)
        return float(structural_similarity(
            orig, recon, channel_axis=-1, data_range=data_range
        ))
    
    @staticmethod
    def calculate_accuracy(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cpu"
    ) -> float:
        """
        Calculate model accuracy on a dataset.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader with test data
            device: Device to run evaluation on
        
        Returns:
            Accuracy as a float between 0 and 1
        """
        model = model.to(device)
        model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def calculate_loss(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: str = "cpu"
    ) -> float:
        """
        Calculate average loss on a dataset.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader with test data
            criterion: Loss function
            device: Device to run evaluation on
        
        Returns:
            Average loss value
        """
        model = model.to(device)
        model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
        
        return total_loss / total_samples if total_samples > 0 else 0.0
    
    @staticmethod
    def evaluate_all(
        original: Optional[Union[torch.Tensor, np.ndarray]] = None,
        reconstructed: Optional[Union[torch.Tensor, np.ndarray]] = None,
        model: Optional[torch.nn.Module] = None,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        criterion: Optional[torch.nn.Module] = None,
        device: str = "cpu",
        metrics: Optional[List[str]] = None,
        data_range: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate multiple metrics at once.
        
        Args:
            original: Original image (for PSNR/SSIM)
            reconstructed: Reconstructed image (for PSNR/SSIM)
            model: Model (for accuracy/loss)
            dataloader: DataLoader (for accuracy/loss)
            criterion: Loss function (for loss metric)
            device: Device to run evaluation on
            metrics: List of metrics to calculate ['psnr', 'ssim', 'accuracy', 'loss']
            data_range: Data range for image metrics
        
        Returns:
            Dictionary with metric names as keys and values
        """
        if metrics is None:
            metrics = ['psnr', 'ssim', 'accuracy', 'loss']
        
        results = {}
        
        if 'psnr' in metrics and original is not None and reconstructed is not None:
            results['psnr'] = MetricsCalculator.calculate_psnr(
                original, reconstructed, data_range
            )
        
        if 'ssim' in metrics and original is not None and reconstructed is not None:
            results['ssim'] = MetricsCalculator.calculate_ssim(
                original, reconstructed, data_range
            )
        
        if 'accuracy' in metrics and model is not None and dataloader is not None:
            results['accuracy'] = MetricsCalculator.calculate_accuracy(
                model, dataloader, device
            )
        
        if 'loss' in metrics and model is not None and dataloader is not None and criterion is not None:
            results['loss'] = MetricsCalculator.calculate_loss(
                model, dataloader, criterion, device
            )
        
        return results

