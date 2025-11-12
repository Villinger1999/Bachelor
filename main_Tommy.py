import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.central_model import get_model
from attack_function.iDLG_Tommy import iDLG
from torchvision import utils as vutils
import numpy as np
from models.lenet import LeNet

# model = get_model()
model = LeNet()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
leaked_grads = torch.load("state_dicts/local_grads_client0_5c_5r_b8_lenet.pt", map_location=torch.device('cpu'), weights_only=True)
grads_dict = leaked_grads["grads"]
grads_list = [v for v in grads_dict.values() if isinstance(v, torch.Tensor)]
label = torch.tensor([leaked_grads['labels'][0].item()], dtype=torch.long, device=device)
dummy_img = torch.randn(1, 3, 32, 32, device=device, dtype=torch.float32)

# Average gradients across batch dimension
avg_grads_list = [g / 8 for g in grads_list]

attacker = iDLG(
    model=model,
    orig_img=dummy_img,      
    label=label,
    device=device,
    grads=avg_grads_list,
    seed=10,             
    clamp=(0.0, 1.0),   
)

recon, pred_label, history, losses = attacker.attack(iterations=50)

print(f"label: {label}, pred_label: {pred_label}")

# Ensure recon_tensor is a torch tensor on CPU with shape (B, C, H, W)
recon_tensor = torch.from_numpy(recon) if isinstance(recon, np.ndarray) else recon
recon_tensor = recon_tensor.detach().cpu()

# Extract first image in batch
img0 = recon_tensor[0]  # (C, H, W)

# Handle both 3-D (C, H, W) and 2-D (H, W) images
if img0.dim() == 3:
    # RGB/multi-channel: (C, H, W) -> (H, W, C)
    img_display = img0.permute(1, 2, 0).numpy()
elif img0.dim() == 2:
    # Grayscale: (H, W)
    img_display = img0.numpy()
else:
    raise ValueError(f"Unexpected image shape: {img0.shape}")

print(f"img0.shape={img0.shape}")

# Create subplot with image and loss
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot reconstructed image
axes[0].imshow(img_display)
axes[0].set_title(f"Reconstructed Image (Pred Label: {pred_label.item()})")
axes[0].axis('off')

# Plot loss curve
axes[1].plot(losses)
axes[1].set_title("Reconstruction Loss")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Loss")
axes[1].grid()

plt.tight_layout()
plt.savefig("reconstruction_result_5c_5r_b8_lenet.png", dpi=100)
plt.show()

print(f"Reconstructed image shape: {recon_tensor.shape}")
print(f"Predicted label: {pred_label}")
print(f"Final loss: {losses[-1]:.6f}")

# problemer med dimensionerne og batch size - læs op på iDLG