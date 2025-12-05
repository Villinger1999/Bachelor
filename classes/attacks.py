from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim.lbfgs import LBFGS
from PIL import Image
from classes.defenses import *
from classes.noise import *

class iDLG:
    def __init__(
        self,
        model,
        label,
        seed: int,
        clamp: tuple[float, float],
        device="cpu",
        orig_img=None,
        grads=None,
        defense=None,
        random_dummy=True,
        dummy_var=0.0,
    ) -> None:
        # Respect provided device and keep original dtype of the model/weights
        self.device = device if isinstance(device, str) else (device.type if hasattr(device, "type") else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='sum').to(self.device)
        self.label = label.to(self.device)
        self.tt = transforms.ToPILImage()
        self.clamp = clamp
        self.defense = defense
        self.grads = grads
        self.var=dummy_var
        self.random_dummy = random_dummy
        self.param_dtype = next(self.model.parameters()).dtype
        if orig_img is not None:
            self.orig_img = orig_img.to(self.device)
            # Align image dtype to model parameter dtype (usually float32)
            if self.orig_img.dtype != self.param_dtype:
                self.orig_img = self.orig_img.to(self.param_dtype)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def attack(self, iterations=200):
        # iDLG training image reconstruction:
        self.model.eval()
        if self.orig_img is not None and self.grads is None:
            # compute original gradients
            predicted = self.model(self.orig_img)
            loss = self.criterion(predicted, self.label)
            orig_grads = torch.autograd.grad(loss, self.model.parameters())
            orig_grads = list((_.detach().clone() for _ in orig_grads))
        elif self.grads is not None:
            orig_grads = self.grads
        else:
            raise ValueError("orig_img and grads cannot both be None")
        
        if self.defense != None:
            orig_grads = self.defense.apply(orig_grads)
            
        
        if self.random_dummy == True:
            dummy_data = (torch.randn(self.orig_img.size(), dtype=self.param_dtype, device=self.device).requires_grad_(True))
        elif self.random_dummy == False and self.orig_img is None:
            dummy_data = (torch.randn(self.orig_img.size(), dtype=self.param_dtype, device=self.device).requires_grad_(True))
        else:
            dummy_data = NoiseGenerator.apply_torch_noise(var=self.var, orig_img=self.orig_img.to(self.device, dtype=self.param_dtype))
            
        dummy_save = dummy_data.detach().cpu().clone()

        # init with ground truth:
        label_pred = torch.argmin(torch.sum(orig_grads[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
        
        optimizer = LBFGS(
            [dummy_data], lr=.1, max_iter=50,
            tolerance_grad=1e-09, tolerance_change=1e-11,
            history_size=100, line_search_fn='strong_wolfe'
        )

        history = []
        losses = []

        for iters in tqdm(range(iterations)):
            def closure():
                optimizer.zero_grad()
                dummy_pred = self.model(dummy_data)
                dummy_loss = self.criterion(dummy_pred, label_pred)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, orig_grads):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)

            # Optional: keep dummy within valid input range
            if self.clamp is not None:
                with torch.no_grad():
                    dummy_data.clamp_(self.clamp[0], self.clamp[1])

            if iters % 1 == 0:
                current_loss = closure()
                losses.append(current_loss.item())
                history.append(self.tt(dummy_data[0].detach().cpu()))
                
        return dummy_save.detach().cpu(), dummy_data.detach().numpy().squeeze(), label_pred, history, losses 
    
def visualize(orig_img, dummy, recon, pred_label, label, losses, random_dummy, dummy_var, grads_mode, var_str, save_name: str):
    """
    orig_img: (1,C,H,W) or (C,H,W) torch tensor
    dummy: (1,C,H,W) or (C,H,W) torch tensor (initial dummy)
    recon: numpy or torch (C,H,W) or (1,C,H,W) – final reconstruction
    """
    recon_tensor = torch.from_numpy(recon) if isinstance(recon, np.ndarray) else recon
    recon_tensor = recon_tensor.detach().cpu()

    if recon_tensor.dim() == 4:
        rec_img = recon_tensor[0]  # (C,H,W)
    elif recon_tensor.dim() == 3:
        rec_img = recon_tensor
    else:
        raise ValueError(f"Unexpected recon tensor shape: {recon_tensor.shape}")

    recon_display = rec_img.permute(1, 2, 0).numpy()  # (H,W,C)

    dummy_tensor = dummy.detach().cpu()
    if dummy_tensor.dim() == 4:
        dummy_img = dummy_tensor[0]
    elif dummy_tensor.dim() == 3:
        dummy_img = dummy_tensor
    else:
        raise ValueError(f"Unexpected dummy tensor shape: {dummy_tensor.shape}")

    dummy_display = dummy_img.permute(1, 2, 0).numpy()

    orig = orig_img.detach().cpu()
    if orig.dim() == 4:
        orig_img0 = orig[0]  # (C,H,W)
    elif orig.dim() == 3:
        orig_img0 = orig
    else:
        raise ValueError(f"Unexpected original image tensor shape: {orig.shape}")

    orig_display = orig_img0.permute(1, 2, 0).numpy()

    # Make a subplot with the dummy data, reconstructed image, original image and the loss curve
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # Dummy initialization
    axes[0].imshow(dummy_display)
    axes[0].set_title(f"Dummy Init\n(random={random_dummy}, var={dummy_var})")
    axes[0].axis("off")

    # Reconstructed image
    axes[1].imshow(recon_display)
    axes[1].set_title(f"Reconstructed\n(Pred Label: {pred_label.item()})")
    axes[1].axis("off")

    # Original CIFAR image
    axes[2].imshow(orig_display)
    axes[2].set_title(f"Original Image\n(Label: {label.item()})")
    axes[2].axis("off")

    # Loss curve
    axes[3].plot(losses)
    axes[3].set_title("Loss curve")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("Loss")
    axes[3].grid(True)

    grads_desc = "Original Gradients" if grads_mode == "none" else f"Leaked Gradients {var_str}"
    fig.suptitle(
        f"iDLG Attack — Model: LeNet — Dummy Strategy: {grads_desc}",
        fontsize=18,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(save_name, dpi=100)
    plt.show()