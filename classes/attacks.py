from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim.lbfgs import LBFGS
from classes.defenses import Defense, SGP, Clipping, clipping_threshold, pruning_threshold
from classes.noise import NoiseGenerator
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod


class Attack(ABC):
    """Base class for all attacks."""
    
    @abstractmethod
    def attack(self, *args, **kwargs):
        """Execute the attack."""
        pass


class iDLG(Attack):
    """iDLG (inverting Deep Learning from Gradients) attack implementation."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        label: torch.Tensor,
        seed: Optional[int] = None,
        clamp: Tuple[float, float] = (0.0, 1.0),
        device: str = "cpu",
        orig_img: Optional[torch.Tensor] = None,
        grads: Optional[List[torch.Tensor]] = None,
        defense: Optional[Defense] = None,
        random_dummy: bool = True,
        dummy_var: float = 0.0,
    ) -> None:
        """
        Initialize iDLG attack.
        
        Args:
            model: Target model
            label: True label (or will be inferred from gradients)
            seed: Random seed
            clamp: Clamp range for reconstructed image
            device: Device to run on
            orig_img: Original image (optional, for computing gradients)
            grads: Leaked gradients (optional, if None will compute from orig_img)
            defense: Defense to apply to gradients (optional)
            random_dummy: Whether to use random dummy initialization
            dummy_var: Variance for noisy dummy initialization
        """
        self.device = device if isinstance(device, str) else (device.type if hasattr(device, "type") else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='sum').to(self.device)
        self.label = label.to(self.device)
        self.tt = transforms.ToPILImage()
        self.clamp = clamp
        self.defense = defense
        self.grads = grads
        self.var = dummy_var
        self.random_dummy = random_dummy
        self.param_dtype = next(self.model.parameters()).dtype
        
        if orig_img is not None:
            self.orig_img = orig_img.to(self.device)
            if self.orig_img.dtype != self.param_dtype:
                self.orig_img = self.orig_img.to(self.param_dtype)
        else:
            self.orig_img = None

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def attack(self, iterations: int = 200) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List, List[float], Optional[List[torch.Tensor]]]:
        """
        Execute iDLG attack.
        
        Args:
            iterations: Number of optimization iterations
        
        Returns:
            Tuple of (dummy_init, reconstructed_image, predicted_label, history, losses, reconstructed_grads)
            reconstructed_grads: Gradients computed from the reconstructed image (for use in FedAvg)
        """
        self.model.eval()
        
        # Get gradients
        if self.orig_img is not None and self.grads is None:
            # Compute original gradients from image
            predicted = self.model(self.orig_img)
            loss = self.criterion(predicted, self.label)
            orig_grads = torch.autograd.grad(loss, self.model.parameters())
            orig_grads = list((_.detach().clone() for _ in orig_grads))
        elif self.grads is not None:
            orig_grads = self.grads
        else:
            raise ValueError("orig_img and grads cannot both be None")
        
        # Apply defense if provided
        if self.defense is not None:
            orig_grads = self.defense.apply(orig_grads)
        
        # Initialize dummy data
        if self.random_dummy:
            if self.orig_img is None:
                raise ValueError("orig_img required for dummy initialization")
            dummy_data = torch.randn(
                self.orig_img.size(), 
                dtype=self.param_dtype, 
                device=self.device
            ).requires_grad_(True)
        else:
            if self.orig_img is None:
                raise ValueError("orig_img required for noisy dummy initialization")
            dummy_data = NoiseGenerator.apply_torch_noise(
                var=self.var, 
                orig_img=self.orig_img.to(self.device, dtype=self.param_dtype)
            )
        
        dummy_save = dummy_data.detach().cpu().clone()

        # Infer label from gradients
        label_pred = torch.argmin(
            torch.sum(orig_grads[-2], dim=-1), dim=-1
        ).detach().reshape((1,)).requires_grad_(False)
        
        # Setup optimizer
        optimizer = LBFGS(
            [dummy_data], lr=1, max_iter=50,
            tolerance_grad=1e-09, tolerance_change=1e-11,
            history_size=100, line_search_fn='strong_wolfe'
        ) 

        history = []
        losses = []

        # Optimization loop
        for iters in tqdm(range(iterations)):
            def closure():
                optimizer.zero_grad()
                dummy_pred = self.model(dummy_data)
                dummy_loss = self.criterion(dummy_pred, label_pred)
                dummy_dy_dx = torch.autograd.grad(
                    dummy_loss, self.model.parameters(), create_graph=True
                )
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, orig_grads):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)

            # Clamp to valid range
            if self.clamp is not None:
                with torch.no_grad():
                    dummy_data.clamp_(self.clamp[0], self.clamp[1])

            if iters % 1 == 0:
                current_loss = closure()
                losses.append(current_loss.item())
                history.append(self.tt(dummy_data[0].detach().cpu()))
                
        # Compute final gradients from reconstructed image (for use in FedAvg)
        self.model.eval()
        reconstructed_grads = None
        if dummy_data is not None:
            # Compute gradients from the reconstructed image
            dummy_pred = self.model(dummy_data.detach())
            dummy_loss = self.criterion(dummy_pred, label_pred)
            reconstructed_grads = torch.autograd.grad(
                dummy_loss, self.model.parameters(), retain_graph=False
            )
            reconstructed_grads = [g.detach().clone() for g in reconstructed_grads]
        
        return (
            dummy_save.detach().cpu(), 
            dummy_data.detach().cpu(), 
            label_pred, 
            history, 
            losses,
            reconstructed_grads  # New: gradients from reconstructed image
        ) 