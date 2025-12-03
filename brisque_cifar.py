import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from test_train_noiseadd import x_test, x_train
import os
import pyiqa
import zipfile
from PIL import Image
from skimage import util, io
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

resolution = 32  # Image resolution (currently unused)

path = os.getcwd() + "/"  # Folder you are in

# setup brisque model
model = pyiqa.create_metric('brisque', device="cpu")

# variance
variances = [0, 0.01, 0.05]

results = []

image_count = 100

images = []
for i in range(image_count):
    images.append(x_train[i])

# Make sure output dirs exist (avoid FileNotFoundError)
os.makedirs(path + 'data/imagenetSubNoise', exist_ok=True)
os.makedirs(path + 'results', exist_ok=True)

# Process images
for idx, img in enumerate(images):
    # img is a numpy array from x_train

    image_array = np.array(img)

    # If values are in [0,1] floats, scale to [0,255] uint8
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).clip(0, 255).astype(np.uint8)

    # If channels-first (C, H, W), convert to channels-last (H, W, C)
    if image_array.ndim == 3 and image_array.shape[0] in (1, 3) and image_array.shape[0] < image_array.shape[-1]:
        image_array = np.transpose(image_array, (1, 2, 0))

    # Base PIL image
    processed_image = Image.fromarray(image_array)

    for variance in variances:
        if variance == 0:
            save_array = image_array
        else:
            noisy_array = util.random_noise(image_array, mode='gaussian', var=variance)
            noisy_array = (noisy_array * 255).clip(0, 255).astype(np.uint8)
            processed_image = Image.fromarray(noisy_array)
            save_array = noisy_array

        # Save noisy versions only for the first image
        if idx == 0:
            io.imsave(path + f'data/imagenetSubNoise/noisy{idx}_{variance}.jpg', save_array)

        brisque_score = model(processed_image).item()
        results.append(
            {
                "image_idx": idx,
                "variance": variance,
                "brisque_score": brisque_score
            }
        )


df_results = pd.DataFrame(results)

# Plot BRISQUE histograms per variance
for variance in variances:
    var_scores = df_results[df_results['variance'] == variance]['brisque_score']
    plt.hist(var_scores, bins=20, alpha=0.6, label=f'var={variance}')

plt.xlabel('BRISQUE Score')
plt.ylabel('Frequency')
plt.title('BRISQUE Scores by Noise Variance')
plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
plt.grid(True, alpha=0.3)

plt.savefig(path + 'results/brisque_analysis3.png', dpi=800, bbox_inches='tight')
plt.close()
