from abc import ABC, abstractmethod
import torch
import numpy as np

class Defense(ABC):
    """Base class all defenses must inherit from."""

    @abstractmethod
    def apply(self, grads):
        """
        Apply the defense to a list of gradients.
        Returns modified gradients.
        """
        pass

class Clipping(Defense):
    """Clips gradients to a maximum absolute magnitude."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def apply(self, grads):
        """
        clipping:
        Set gradients with magnitude >= threshold to threshold.

        Args:
            model (nn.Module): model with gradients
            threshold (float): threshold for cutoff
        """
        return [grad.clamp(-self.threshold, self.threshold) for grad in grads]

class SGP(Defense):
    """Small Gradient Pruning: zero out small gradients."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def apply(self, grads):
        """
        Small Gradient Pruning (sgp):
        Set gradients with magnitude <= threshold to 0.

        Args:
            model (nn.Module): model with gradients
            threshold (float): threshold for cutoff
        """   
        pruned = []
        for grad in grads:
            mask = grad.abs() <= self.threshold
            pruned.append(torch.where(mask, torch.zeros_like(grad), grad))
        return pruned
    
def clipping_threshold(grads, percentile: float = 0.9):
    abs_vals = torch.cat([g.detach().abs().reshape(-1) for g in grads if g is not None])
    return float(np.percentile(abs_vals, percentile))


def pruning_threshold(grads, keep_ratio: float = 0.9):
    assert 0.0 < keep_ratio <= 1.0
    abs_vals = torch.cat([g.detach().abs().reshape(-1) for g in grads if g is not None])
    return float(torch.quantile(abs_vals, 1.0 - keep_ratio))
