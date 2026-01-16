from abc import ABC, abstractmethod
import torch
import numpy as np
from collections import deque

class Defense(ABC):
    """Base class all defenses must inherit from."""

    @abstractmethod
    def apply(self, grads):
        """
        Apply the defense to a list of gradients.
        Returns modified gradients.
        """
        pass

class NormClipping:
    """
    Layerwise norm clipping.

    Steps:
      1) compute L2 norm per gradient tensor
      2) tau = quantile(norms, q)  (q close to 1 keeps most layers unclipped)
      3) for each layer: g <- g * min(1, tau / (||g|| + eps))
    """
    def __init__(self, q: float = 0.9, eps: float = 1e-6):
        assert 0.0 < q < 1.0
        self.q = q
        self.eps = eps

    @torch.no_grad()
    def apply(self, grads: list[torch.Tensor]):
        # norms for non-None grads only
        norms = []
        idxs = []
        for i, g in enumerate(grads):
            if g is None:
                continue
            n = g.norm(p=2)
            norms.append(n)
            idxs.append(i)

        if len(norms) == 0:
            return grads

        norms_t = torch.stack(norms) 
        tau = torch.quantile(norms_t, self.q)

        out = list(grads)
        # Clip each layer independently to tau
        for n, i in zip(norms_t, idxs):
            scale = min(1.0, (tau / (n + self.eps)).item())
            out[i] = out[i] * scale
        return out

class Clipping(Defense):
    """Clips gradients to a maximum absolute magnitude."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def apply(self, grads):
        """
        Value clipping:
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
    return float(torch.quantile(abs_vals, percentile))


def pruning_threshold(grads, keep_ratio: float = 0.9):
    assert 0.0 < keep_ratio <= 1.0
    abs_vals = torch.cat([g.detach().abs().reshape(-1) for g in grads if g is not None])
    return float(torch.quantile(abs_vals, 1.0 - keep_ratio))
