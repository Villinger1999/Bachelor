import torch
import numpy as np
import matplotlib.pyplot as plt 
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def fix_dimension(img):
    """
    Converts a PyTorch tensor to a NumPy array safely.
    Handles GPU tensors, gradients, and batch dims.
    Ensures the array is (H,W,C) for color images.
    Handles (1,C,H,W), (C,H,W), or already (H,W,C).
    """
    if isinstance(img, np.ndarray):
        arr = img
    else:
        arr = img.detach().cpu().numpy()
    
        # (1,C,H,W) -> (C,H,W)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr.squeeze(0)

    # (C,H,W) -> (H,W,C)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    
    return arr

def visualize(orig_img, dummy, recon, pred_label, label, losses, random_dummy, dummy_var, grads_mode, var_str, save_name: str):
    """
    orig_img: (1,C,H,W) or (C,H,W) torch tensor
    dummy: (1,C,H,W) or (C,H,W) torch tensor (initial dummy)
    recon: numpy or torch (C,H,W) or (1,C,H,W) - final reconstruction
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
    
def apply_defended_grads(model, defended_grads, lr=0.01, momentum=0.9):
    """
    Apply a single SGD update using a list of defended gradients aligned with model.parameters().
    """
    if defended_grads is None:
        return  

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    optimizer.zero_grad(set_to_none=True)

    for p, g in zip(model.parameters(), defended_grads):
        if g is None:
            p.grad = None
        else:
            p.grad = g.detach().to(device=p.device, dtype=p.dtype)

    optimizer.step()

def tensor_to_hwc01(x: torch.Tensor) -> np.ndarray:
    if x.dim() == 4:
        x = x[0]
    x = x.detach().cpu()
    x = x.permute(1, 2, 0)  # (H,W,C)
    return x.numpy()

def compute_ssim_psnr(true_img: torch.Tensor, recon_img: torch.Tensor):
    a = tensor_to_hwc01(true_img)
    b = tensor_to_hwc01(recon_img)
    ssim = structural_similarity(a, b, channel_axis=-1, data_range=1.0)
    psnr = peak_signal_noise_ratio(a, b, data_range=1.0)
    return float(ssim), float(psnr)
