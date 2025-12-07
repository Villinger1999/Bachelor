from abc import ABC, abstractmethod

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
        clipped_grads = []
        for grad in grads:      
            if grad.abs() >= self.threshold:
                clipped = grad.clamp(min=-self.threshold, max=self.threshold)
                clipped_grads.append(clipped)  # scale down large grads
            else:
                clipped_grads.append(grad)
        return clipped_grads


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
        pruned_grads = []
        for grad in grads:      
            if grad.abs() <= self.threshold:
                pruned_grads.append(0.0)  # scale down large grads
            else:
                pruned_grads.append(grad)
        return pruned_grads


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
        pruned_grads = []
        for grad in grads:      
            if grad.abs() > self.threshold:
                pruned_grads.append(self.alpha * grad)  # scale down large grads
            else:
                pruned_grads.append(grad)
        return pruned_grads
