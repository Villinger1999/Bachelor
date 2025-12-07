import torch
import numpy as np
import matplotlib.pyplot as plt 

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