import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.central_model import get_model

model = get_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

attacker = iDLG(
    model=model,
    orig_img=img,       # We don't have the original image or the label???
    label=label,
    device=device,
    seed=0,             
    clamp=(0.0, 1.0),   
)

recon, pred_label, history, losses = attacker.attack(iterations=50)
