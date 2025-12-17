from abc import ABC, abstractmethod
import torch

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

class PLGP(Defense):
    """Proportional Large Gradient Pruning."""

    def __init__(self, threshold: float, alpha: float):
        self.threshold = threshold
        self.alpha = alpha

    def apply(self, grads):
        """
        Proportional Large Gradient Pruning (PLGP):
        Scales down gradients with magnitude > threshold by factor alpha.

        Args:
            model (nn.Module): model with gradients
            threshold (float): threshold for cutoff
            alpha (float): scaling factor for large gradients (0 < alpha < 1)
        """
        out = []
        for grad in grads:
            mask = grad.abs() > self.threshold
            out.append(torch.where(mask, self.alpha * grad, grad))
        return out

