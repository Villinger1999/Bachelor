import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pyiqa
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import os
import time
from test_train_noiseadd import resolution


# Make path the folder path
path = os.getcwd() + "/"   # always points to the folder you are in

# Setup brisque model
model = pyiqa.create_metric('brisque', device='cpu')

results = []

# Create array of image path
var = ['noisy0_0.jpg', 'noisy0_0.01.jpg', 'noisy0_0.05.jpg', 'noisy0_0.1.jpg', 'noisy0_0.2.jpg', 'noisy0_0.3.jpg', 'noisy0_0.4.jpg', 'noisy0_0.5.jpg']
var1 = ['noisy1_0.jpg', 'noisy1_0.01.jpg', 'noisy1_0.05.jpg', 'noisy1_0.1.jpg', 'noisy1_0.2.jpg', 'noisy1_0.3.jpg', 'noisy1_0.4.jpg', 'noisy1_0.5.jpg']
var2 = ['noisy2_0.jpg', 'noisy2_0.01.jpg', 'noisy2_0.05.jpg', 'noisy2_0.1.jpg', 'noisy2_0.2.jpg', 'noisy2_0.3.jpg', 'noisy2_0.4.jpg', 'noisy2_0.5.jpg']

# Reference images
ref0 = cv2.cvtColor(cv2.imread(path + "data/noise/" + var[0]), cv2.COLOR_BGR2RGB)
ref1 = cv2.cvtColor(cv2.imread(path + "data/noise/" + var1[0]), cv2.COLOR_BGR2RGB)
ref2 = cv2.cvtColor(cv2.imread(path + "data/noise/" + var2[0]), cv2.COLOR_BGR2RGB)

def evaluate_set(file_list, ref_img, img_id):
    for fname in file_list:
        noisy = cv2.cvtColor(cv2.imread(path + "data/noise/" + fname), cv2.COLOR_BGR2RGB)
        variance = float(fname.split("_")[-1].replace(".jpg", ""))
        noisy_pil = Image.fromarray(noisy)

        row = {"Image": img_id, "Variance": variance}

        # NR-IQA models
        row["BRISQUE"] = model(noisy_pil).item()

        # FR-IQA metrics
        row["PSNR"] = psnr(ref_img, noisy, data_range=255)
        row["SSIM"] = ssim(ref_img, noisy, channel_axis=-1, data_range=255)

        results.append(row)

# Run evaluations
evaluate_set(var, ref0, "img0")
evaluate_set(var1, ref1, "img1")
evaluate_set(var2, ref2, "img2")

# DataFrame
df_metrics = pd.DataFrame(results).sort_values(by=["Image", "Variance"])

# Plot scores vs variance
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
        ax.plot(group["Variance"], group[metric], marker="o", label=img_id)
    ax.set_title(metric)
    ax.set_xlabel("Gaussian Noise Variance")
    ax.set_ylabel(ylabels[metric])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

plt.suptitle(f"Image Quality Metrics vs Noise Variance res: {resolution}", fontsize=16, y=1)
plt.savefig(path + "results/assessment/brisque_results_" + str(time.time_ns()) + ".png", dpi=150)
df_metrics.to_csv(path + "results/assessment/brisque_" + str(time.time_ns()) + ".csv", index=False)



