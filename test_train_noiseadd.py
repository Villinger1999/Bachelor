import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import tensorflow as tf
from skimage import io, util
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# Make path the folder path
path = os.getcwd() + "/"   

# Download CIFAR-10 and load it into memory
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Train:", x_train.shape, y_train.shape)
print("Test:", x_test.shape, y_test.shape)

# Variances for Gaussian noise
variances = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

x = 256 # Resolution
resize = transforms.Resize((x, x))

# Store results
results = []
noisy_img = []

image = Image.open('data/hest.jpg')  
img_resized = resize(image)                  
img_resized = np.array(img_resized) / 255.0  #normalize

for var in variances:
    noisy0 = util.random_noise(x_train[0], mode='gaussian', var=var)
    noisy1 = util.random_noise(x_train[1], mode='gaussian', var=var)
    noisy2 = util.random_noise(img_resized, mode='gaussian', var=var) # High resolution image
    
    # Save images
    fname0 = f"noisy0_{var}.jpg"
    fname1 = f"noisy1_{var}.jpg"
    fname2 = f"noisy2_{var}.jpg"
    
    io.imsave(path + 'data/noise/' + fname0, (noisy0 * 255).astype("uint8"))
    io.imsave(path + 'data/noise/' + fname1, (noisy1 * 255).astype("uint8"))
    io.imsave(path + 'data/noise/' + fname2, (noisy2 * 255).astype("uint8")) # High resolution image

    
    # Append metadata for DataFrame
    results.append({"Image": "img0", "Variance": var, "Path": fname0, "Array": noisy0})
    results.append({"Image": "img1", "Variance": var, "Path": fname1, "Array": noisy1})
    results.append({"Image": "img2", "Variance": var, "Path": fname2, "Array": noisy2}) # High resolution image


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
show_noisy_grid(df[df["Image"] == "img2"].to_dict("records"), "Image 2", path + "data/noise/range_noise2.jpg")
