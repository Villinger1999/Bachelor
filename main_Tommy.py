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
import matplotlib.image as mpimg

# model = get_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
leaked_grads = torch.load("state_dicts/local_grads_client0_c1_b1_e1_emptyLeNet.pt", map_location=torch.device('cpu'), weights_only=True)
model = LeNet()
state = leaked_grads["model_state"]
model.load_state_dict(state)
model = model.to(device)
grads_dict = leaked_grads["grads_per_sample"]
grads_list = [grads_dict[name] for name, _ in model.named_parameters()]
# grads_list = [v for v in grads_dict.values() if isinstance(v, torch.Tensor)]
label = torch.tensor([leaked_grads['labels_per_sample'].item()], dtype=torch.long, device=device)
dummy_img = torch.randn(1, 3, 32, 32, device=device, dtype=torch.float32)

attacker = iDLG(
    model=model,
    orig_img=dummy_img,      
    label=label,
    device=device,
    grads=grads_list,
    seed=10,             
    clamp=(0.0, 1.0),   
)

recon, pred_label, history, losses = attacker.attack(iterations=100)

# Ensure recon_tensor is a torch tensor on CPU with shape (B, C, H, W)
recon_tensor = torch.from_numpy(recon) if isinstance(recon, np.ndarray) else recon
recon_tensor = recon_tensor.detach().cpu()

# Extract first image in batch
img0 = recon_tensor  # (C, H, W)

# Handle both 3-D (C, H, W) and 2-D (H, W) images
if img0.dim() == 3:
    # RGB/multi-channel: (C, H, W) -> (H, W, C)
    img_display = img0.permute(1, 2, 0).numpy()
elif img0.dim() == 2:
    # Grayscale: (H, W)
    img_display = img0.numpy()
else:
    raise ValueError(f"Unexpected image shape: {img0.shape}")

test_img = mpimg.imread("test.png")   # expects H×W×C format

# Create subplot with 3 panels: reconstructed, test image, and loss curve
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Reconstructed image
axes[0].imshow(img_display)
axes[0].set_title(f"Reconstructed Image\n(Pred Label: {pred_label.item()})")
axes[0].axis("off")

# External image: test.png
axes[1].imshow(test_img)
axes[1].set_title(f"Original Test Image\n(Label: {label}")
axes[1].axis("off")

# Loss curve
axes[2].plot(losses)
axes[2].set_title("Reconstruction Loss")
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel("Loss")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("reconstruction_33_adam.png", dpi=100)
plt.show()

print(f"Reconstructed image shape: {recon_tensor.shape}")
print(f"Predicted label: {pred_label}")
print(f"Final loss: {losses[-1]:.6f}")

# problemer med dimensionerne og batch size - læs op på iDLG