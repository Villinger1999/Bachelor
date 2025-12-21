import sys
import torch
from classes.attacks import iDLG
from classes.defenses import *
from classes.models import LeNet
import tensorflow as tf
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from classes.helperfunctions import *

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LeNet()
model.load_state_dict(torch.load("state_dicts/state_dict_model_b64_e150.pt", map_location=device, weights_only=True))
# model.load_state_dict(torch.load("state_dicts/global_state_exp1_c6_b64_e10_FL.pt", map_location=device, weights_only=True))
model = model.to(device)

leaked_grads = torch.load(
    "state_dicts/local_grads_client0_c1_b1_e1_pretrained.pt",
    map_location=torch.device('cpu'),
    weights_only=True
)
grads_dict = leaked_grads["grads_per_sample"]
grads_list = [v for v in grads_dict.values() if isinstance(v, torch.Tensor)]

grads_mode = sys.argv[1].lower()
dummy_str = sys.argv[2].lower()
var_str = sys.argv[3]
img_idx = sys.argv[4]
defense_input = sys.argv[5]

# Map grads_mode → actual gradients parameter
if grads_mode == "none":
    grads = None
elif grads_mode == "grads_list":
    grads = grads_list
else:
    raise ValueError(f"Unknown grads_mode: {grads_mode}")

# String → bool for random dummy
if dummy_str in ("true", "1", "yes"):
    random_dummy = True
elif dummy_str in ("false", "0", "no"):
    random_dummy = False
else:
    raise ValueError(f"random_dummy must be true/false, got {dummy_str}")

# String → float for variance
dummy_var = float(var_str)
idx = int(img_idx)
 
# use true label from CIFAR-10
label_value = int(y_train[idx][0])
label = torch.tensor([label_value], dtype=torch.long, device=device)

# x_train[0] is (32,32,3) in [0,255]
orig_np = x_train[idx].astype("float32") / 255.0  # -> (H,W,C) in [0,1]
orig_tensor = torch.from_numpy(orig_np)        # (H,W,C)
orig_tensor = orig_tensor.permute(2, 0, 1)     # -> (C,H,W)

# ensure batch dimension: (1,C,H,W)
orig_img = orig_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)

attacker = iDLG(
    model=model,
    label=label,
    seed=None,
    clamp=(0.0, 1.0),
    device=device,
    orig_img=orig_img,
    grads=grads,
    defense=defense_input,
    random_dummy=random_dummy,
    dummy_var=dummy_var,
)

defense_save, dummy, recon, pred_label, history, losses = attacker.attack(iterations=100)

print(f"Predicted label: {pred_label}")
print(f"Final loss: {losses[-1]:.6f}")
print(f"Random dummy: {random_dummy}, dummy variance: {dummy_var}, grads_mode: {grads_mode}")

visualize(
    orig_img=orig_img,
    dummy=dummy,
    recon=recon,
    pred_label=pred_label,
    label=label,
    losses=losses,
    random_dummy=random_dummy,
    dummy_var=dummy_var,
    grads_mode=grads_mode,
    var_str=var_str,
    save_name=f"reconstruction_{sys.argv[7]}.png",
)

imTrue = fix_dimension(orig_img)

imRecon = fix_dimension(recon)

ssim_val = structural_similarity(imTrue, imRecon, channel_axis=-1, data_range=1.0)
psnr_val = peak_signal_noise_ratio(imTrue, imRecon, data_range=1.0)

print("SSIM:", ssim_val)
print("PSNR:", psnr_val)