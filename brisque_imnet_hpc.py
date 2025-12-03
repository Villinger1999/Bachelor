import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image
from skimage import util, io
import pyiqa
from load_imnet import train    
from torchvision import transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# label map
synsets = sorted({os.path.basename(os.path.dirname(p)) for p in train})
class_to_idx = {s: i for i, s in enumerate(synsets)}

# subset (5%)
total_size = len(train)
subset_size = 100
subset = random.sample(train, subset_size)
print("Train subset size:", subset_size)


img_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize((32, 32)),    # <<--- Reduce resolution
    transforms.ToTensor(),          # (C, 32, 32)
])

x_train = []
y_train = []

for path in subset:
    img = Image.open(path).convert('RGB')
    img = img_tf(img)      # now (3, 32, 32)
    x_train.append(img)

    synset = os.path.basename(os.path.dirname(path))
    label = class_to_idx[synset]
    y_train.append(label)

x_train_torch = torch.stack(x_train)                # (N, 3, 32, 32)
y_train_torch = torch.tensor(y_train, dtype=torch.long)

print("Tensor shape:", x_train_torch.shape)

model = pyiqa.create_metric('brisque', device="cpu")

variances = [0, 0.01, 0.05, 0.1, 0.2] # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1

results = []

image_count = min(100, x_train_torch.size(0))
images = x_train_torch[:image_count]      # (100, 3, 32, 32)

path = os.getcwd() + "/"
os.makedirs(path + 'data/imagenetSubNoise', exist_ok=True)
os.makedirs(path + 'results', exist_ok=True)

for idx, img_t in enumerate(images):
    # (3, 32, 32) → (32, 32, 3)
    img_np = img_t.permute(1, 2, 0).cpu().numpy()

    # [0,1] → uint8
    base_array = (img_np * 255).clip(0, 255).astype(np.uint8)

    for variance in variances:
        if variance == 0:
            noisy_array = base_array
        else:
            noisy = util.random_noise(base_array, mode="gaussian", var=variance)
            noisy_array = (noisy * 255).clip(0, 255).astype(np.uint8)

        processed_image = Image.fromarray(noisy_array)

        # save noisy example of the first image
        if idx == 0:
            io.imsave(
                path + f"data/imagenetSubNoise/imnet_noisy{idx}_{variance}.jpg",
                noisy_array
            )

        score = model(processed_image).item()

        results.append({
            "image_idx": idx,
            "variance": variance,
            "brisque_score": score
        })

df_results = pd.DataFrame(results)
df_results.to_csv(path + "results/brisque_imnet_32x32.csv", index=False)

for variance in variances:
    var_scores = df_results[df_results["variance"] == variance]["brisque_score"]
    plt.hist(var_scores, bins=20, alpha=0.6, label=f"var={variance}")

plt.xlabel("BRISQUE Score")
plt.ylabel("Frequency")
plt.title("BRISQUE Scores by Noise Variance (ImageNet, 32×32)")
plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
plt.grid(True, alpha=0.3)
plt.savefig(path + "results/brisque_imnet_32x32.png", dpi=300, bbox_inches="tight")
plt.close()
