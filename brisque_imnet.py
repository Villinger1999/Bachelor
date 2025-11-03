import os
import random
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pyiqa
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

path = Path.cwd()
subset_path = path / "data" / "imnet"
noise_path = path / "data" / "noise"
results_path = path / "results" / "assessment"

noise_path.mkdir(parents=True, exist_ok=True)
results_path.mkdir(parents=True, exist_ok=True)


image_paths = list(subset_path.rglob("*.jpg")) + list(subset_path.rglob("*.png")) + list(subset_path.rglob("*.jpeg"))
print(f"Found {len(image_paths)} images in subset")

sample_paths = random.sample(image_paths, 3)
print("Sampled images:")
for p in sample_paths:
    print("  -", p.name)


plt.figure(figsize=(12, 4))
for i, img_path in enumerate(sample_paths):
    img = Image.open(img_path)
    plt.subplot(1, 3, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(img_path.name[:20])
plt.tight_layout()
plt.show()


noise_variances = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
var_sets = []  # to hold lists of filenames per image

for idx, img_path in enumerate(sample_paths):
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    fname_base = img_path.stem
    file_list = []

    for var_value in noise_variances:
        noisy_img = img.copy()
        if var_value > 0:
            noise = np.random.normal(0, np.sqrt(var_value), img.shape) # gaussian noise 
            noisy_img = np.clip(img + noise, 0, 1) # add noise and ensure that all pixel values stays between 0 and 1

        noisy_uint8 = (noisy_img * 255).astype(np.uint8)
        fname = f"{fname_base}_noisy_{var_value}.jpg"
        cv2.imwrite(str(noise_path / fname), cv2.cvtColor(noisy_uint8, cv2.COLOR_RGB2BGR))
        file_list.append(fname)

    var_sets.append(file_list)

print("Saved noisy images to:", noise_path)


model = pyiqa.create_metric('brisque', device='cpu')
results = []


def evaluate_set(file_list, ref_img, img_label, original_name):
    for fname in file_list:
        noisy_path = noise_path / fname
        noisy = cv2.cvtColor(cv2.imread(str(noisy_path)), cv2.COLOR_BGR2RGB)
        variance = float(fname.split("_")[-1].replace(".jpg", ""))
        noisy_pil = Image.fromarray(noisy)

        results.append({
            "Image": img_label,
            "OriginalFile": original_name,
            "Variance": variance,
            "BRISQUE": model(noisy_pil).item(),
            "PSNR": psnr(ref_img, noisy, data_range=255),
            "SSIM": ssim(ref_img, noisy, channel_axis=-1, data_range=255)
        })


for idx, file_list in enumerate(var_sets):
    ref_img = cv2.cvtColor(cv2.imread(str(noise_path / file_list[0])), cv2.COLOR_BGR2RGB)
    evaluate_set(file_list, ref_img, f"img{idx}", sample_paths[idx].name)


df_metrics = pd.DataFrame(results).sort_values(by=["Image", "Variance"])


metrics = ["BRISQUE", "PSNR", "SSIM"]
ylabels = {
    "BRISQUE": "Lower=Better",
    "PSNR": "Higher=Better (dB)",
    "SSIM": "Higher=Better"
}

fig, axes = plt.subplots(1, 3, figsize=(22, 5))
axes = axes.ravel()

for ax, metric in zip(axes, metrics):
    for img_id, group in df_metrics.groupby("Image"):
        ax.plot(group["Variance"], group[metric], marker="o", label=group["OriginalFile"].iloc[0])
    ax.set_title(metric)
    ax.set_xlabel("Gaussian Noise Variance")
    ax.set_ylabel(ylabels[metric])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

plt.suptitle(f"Image Quality Metrics vs Noise Variance (res={224})", fontsize=16, y=1.02)

timestamp = str(time.time_ns())
plt.savefig(results_path / f"brisque_results_{timestamp}.png", dpi=150, bbox_inches="tight")
df_metrics.to_csv(results_path / f"brisque_{timestamp}.csv", index=False)

print(f"Saved results to {results_path}")
