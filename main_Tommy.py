import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.central_model import get_model
from attack_function.iDLG_Tommy import iDLG
from torchvision import utils as vutils
import numpy as np

model = get_model()
# use LeNet
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
leaked_grads = torch.load("state_dicts/local_grads_client0_5c_5r_b2_size10.pt", map_location=torch.device('cpu'), weights_only=True)
grads_dict = leaked_grads["grads"]
grads_list = [v for v in grads_dict.values() if isinstance(v, torch.Tensor)]
label = torch.tensor([leaked_grads['labels'][0].item()], dtype=torch.long, device=device)
dummy_img = torch.randn(1, 3, 32, 32, device=device, dtype=torch.float32)

# Average gradients across batch dimension
avg_grads_list = []
for grad_tensor in grads_list:
    # Assuming batch dimension is first: (B, ...)
    avg_grad = grad_tensor.mean(dim=0, keepdim=True)
    avg_grads_list.append(avg_grad)

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

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Ensure recon_tensor is a torch tensor on CPU with shape (B, C, H, W)
recon_tensor = torch.from_numpy(recon) if isinstance(recon, np.ndarray) else recon
recon_tensor = recon_tensor.detach().cpu()

# Option A: show first image in batch
img0 = recon_tensor[0]  # (C, H, W)
axes[0].imshow(img0.permute(1, 2, 0).numpy())
axes[0].set_title(f"Reconstructed (first, pred label: {pred_label})")
axes[0].axis('off')


# grid = vutils.make_grid(recon_tensor, nrow=min(8, recon_tensor.size(0)), normalize=True, pad_value=1)
# axes[0].imshow(grid.permute(1,2,0).numpy())
# axes[0].set_title(f"Reconstructed grid (pred label: {pred_label})")
# axes[0].axis('off')

# Plot loss curve
axes[1].plot(losses)
axes[1].set_title("Reconstruction Loss")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Loss")
axes[1].grid()

plt.tight_layout()
plt.savefig("reconstruction_result2.png", dpi=100)
plt.show()
 
print(f"Reconstructed image shape: {recon_tensor.shape}")
print(f"Predicted label: {pred_label}")