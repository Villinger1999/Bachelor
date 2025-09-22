import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pyiqa
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from torchvision import transforms
import tensorflow as tf
from skimage import io, util
import os

# Make path the folder path
path = os.getcwd() + "/"   # always points to the folder you are in

# Download CIFAR-10 and load it into memory
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Train:", x_train.shape, y_train.shape)
print("Test:", x_test.shape, y_test.shape)

# Variances for Gaussian noise
variances = [0, 0.01, 0.05, 0.1, 0.5]

# Store results
results = []

for var in variances:
    noisy0 = util.random_noise(x_train[0], mode='gaussian', var=var)
    noisy1 = util.random_noise(x_train[1], mode='gaussian', var=var)
    
    # Save images
    fname0 = f"noisy0_{var}.jpg"
    fname1 = f"noisy1_{var}.jpg"
    io.imsave(path + 'data/noise/' + fname0, (noisy0 * 255).astype("uint8"))
    io.imsave(path + 'data/noise/' + fname1, (noisy1 * 255).astype("uint8"))
    
    # Append metadata for DataFrame
    results.append({"Image": "img0", "Variance": var, "Path": fname0, "Array": noisy0})
    results.append({"Image": "img1", "Variance": var, "Path": fname1, "Array": noisy1})

# Convert to DataFrame
df = pd.DataFrame(results)

# Grid visualization of added noise
def show_noisy_grid(img_results, title_prefix="Image", save_path=None):
    fig, axes = plt.subplots(1, len(img_results), figsize=(15, 3))
    for ax, row in zip(axes, img_results):
        ax.imshow(row["Array"])
        ax.set_title(f"{title_prefix}\nvar={row['Variance']}")
        ax.axis("off")
    
    if save_path:
        plt.savefig(save_path)
        
show_noisy_grid(df[df["Image"] == "img0"].to_dict("records"), "Image 0", path + "data/noise/range_noise0.jpg")
show_noisy_grid(df[df["Image"] == "img1"].to_dict("records"), "Image 1", path + "data/noise/range_noise1.jpg")

# Load NR-IQA models
models = {
    "BRISQUE": pyiqa.create_metric('brisque', device='cpu'),
    "NIQE":    pyiqa.create_metric('niqe', device='cpu'),
    "ILNIQE":  pyiqa.create_metric('ilniqe', device='cpu'),
    "MANIQA":  pyiqa.create_metric('maniqa', device='cpu')
}

# Resize images to fit NR-IQA models
resize_small = transforms.Resize((128, 128))   # for classical NR-IQA (≥96x96)
resize_large = transforms.Resize((224, 224))   # for MANIQA (≥224x224)

var = ['noisy0_0.jpg', 'noisy0_0.01.jpg', 'noisy0_0.05.jpg', 'noisy0_0.1.jpg', 'noisy0_0.5.jpg']
var1 = ['noisy1_0.jpg', 'noisy1_0.01.jpg', 'noisy1_0.05.jpg', 'noisy1_0.1.jpg', 'noisy1_0.5.jpg']

# Reference images
ref0 = cv2.cvtColor(cv2.imread(path + "data/noise/" + var[0]), cv2.COLOR_BGR2RGB)
ref1 = cv2.cvtColor(cv2.imread(path + "data/noise/" + var1[0]), cv2.COLOR_BGR2RGB)

results = []

def evaluate_set(file_list, ref_img, img_id):
    for fname in file_list:
        noisy = cv2.cvtColor(cv2.imread(path + "data/noise/" + fname), cv2.COLOR_BGR2RGB)
        variance = float(fname.split("_")[-1].replace(".jpg", ""))
        noisy_pil = Image.fromarray(noisy)

        row = {"Image": img_id, "Variance": variance}

        # NR-IQA models
        row["BRISQUE"] = models["BRISQUE"](noisy_pil).item()
        row["NIQE"]    = models["NIQE"](resize_small(noisy_pil)).item()
        row["ILNIQE"]  = models["ILNIQE"](resize_small(noisy_pil)).item()
        row["MANIQA"]  = models["MANIQA"](resize_large(noisy_pil)).item()

        # FR-IQA metrics
        row["PSNR"] = psnr(ref_img, noisy, data_range=255)
        row["SSIM"] = ssim(ref_img, noisy, channel_axis=-1, data_range=255)

        results.append(row)

# Run evaluations
evaluate_set(var, ref0, "img0")
evaluate_set(var1, ref1, "img1")

# DataFrame
df_metrics = pd.DataFrame(results).sort_values(by=["Image", "Variance"])
df_metrics.to_csv(path + "results/assesment/all_metrics_results.csv", index=False)

# Plot all metrics vs variance
metrics = ["BRISQUE", "NIQE", "ILNIQE", "MANIQA", "PSNR", "SSIM"]
ylabels = {
    "BRISQUE": "Lower=Better",
    "NIQE": "Lower=Better",
    "ILNIQE": "Lower=Better",
    "MANIQA": "Higher=Better",
    "PSNR": "Higher=Better (dB)",
    "SSIM": "Higher=Better"
}

fig, axes = plt.subplots(2, 3, figsize=(22, 10))
axes = axes.ravel()

for ax, metric in zip(axes, metrics):
    for img_id, group in df_metrics.groupby("Image"):
        ax.plot(group["Variance"], group[metric], marker="o", label=img_id)
    ax.set_title(metric)
    ax.set_xlabel("Gaussian Noise Variance")
    ax.set_ylabel(ylabels[metric])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

plt.suptitle("Image Quality Metrics vs Noise Variance", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(path + "results/assesment/all_metrics_results.png", dpi=150)
plt.show()
