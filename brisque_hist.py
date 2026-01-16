import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import pyiqa
from PIL import Image
from skimage import util, io
import numpy as np

# a line that makes the code run
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Use the same path as in brisque_value.py
imagenet_path = "/dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/test/"
image_count = 100

def get_images(image_dir, image_count):
    all_files = [f for f in os.listdir(image_dir) if f.endswith('.JPEG')]
    all_files.sort()  # Ensures deterministic order
    selected_files = all_files[:image_count]
    return [os.path.join(image_dir, fname) for fname in selected_files]

# Make path the folder path
path = os.getcwd() + "/" # always points to the folder you are in

#setup brique model
model = pyiqa.create_metric('brisque', device="cpu")

# variance
variances = [0, 0.00016, 0.0049,0.0081, 0.0144, 0.03, 0.04]

results = []

resolution = 180


# Get image paths using get_images from brisque_value.py
image_paths = get_images(imagenet_path, image_count)

# Process images
for idx, img_path in enumerate(image_paths):
    image = Image.open(img_path)
    image = image.resize((resolution,resolution))
    image_array = np.array(image)
    for round_num in range(1):
        for variance in variances:
            if variance == 0:
                processed_image = Image.fromarray(image_array)
                save_array = image_array
            else:
                # Each round will have a different random noise due to the random seed/state
                noisy_array = util.random_noise(image_array, mode='gaussian', var=variance)
                noisy_array = (noisy_array * 255).astype(np.uint8)
                processed_image = Image.fromarray(noisy_array)
                save_array = noisy_array
            # Save images for the first image only (idx == 0 and round_num == 0)
            # if idx == 0 and round_num == 0:
            #     io.imsave(path + f'data/imagenetSubNoise/noisy{idx}_{variance}.jpg', save_array)
            # Calculate BRISQUE score for each variance level
            brisque_score = model(processed_image).item()
            results.append({"image_idx": idx, "variance": variance, "round": round_num, "brisque_score": brisque_score})

df_results = pd.DataFrame(results)

# Extract just the BRISQUE scores for the histogram
# brisque_scores = df_results['brisque_score'].tolist()
# print(brisque_scores)

for variance in variances:
    var_scores = df_results[df_results['variance'] == variance]['brisque_score']
    plt.hist(var_scores, bins=20, alpha=0.6, label=f'var={variance}')
plt.xlabel('BRISQUE Score')
plt.ylabel('Frequency')
plt.title('BRISQUE Scores by Noise Variance')
plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
plt.grid(True, alpha=0.3)
plt.savefig(path + f'results/brisque_analysis_{resolution}.png', dpi=800, bbox_inches='tight')
plt.close()
