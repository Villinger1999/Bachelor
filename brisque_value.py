import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import pyiqa
import zipfile
from PIL import Image
from skimage import util, io
import numpy as np

#a line that makes the code run
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

resolution = 256  # Image resolution

# Make path the folder path
path = os.getcwd() + "/" # always points to the folder you are in

#setup brique model
model = pyiqa.create_metric('brisque', device="cpu")

# variance
variances = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

results = []

# path to zipped data
zip_path = path + "data/archive.zip"
# path to subset folder
extract_to = path + "data/imagenetSub"
# Number of images to be extracted from the zip folder
image_count = 100

# getting the first 1000 images from the archive folder in data
with zipfile.ZipFile(zip_path,'r') as zip_folder:
    # get all images from the dataset (ImageNet) 
    all_files = [file for file in zip_folder.namelist() if file.lower().endswith('.jpg')]
    # create an list with the names of the first n (image_count) images from the list of file names to the 
    selected_files = all_files[:image_count]
    #extract the first n images from the zipfolder to the subsetfolder 
    zip_folder.extractall(extract_to, members=selected_files)

# function to get the paths to the first n images in a folder 
def get_image_paths(root_dir, max_images=image_count):
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg')):
                image_paths.append(os.path.join(root, file))
                if len(image_paths) >= max_images:
                    return image_paths
    return image_paths

# Get image paths
image_paths = get_image_paths(extract_to, image_count)

# Process images - back to your original simple approach
for idx, img_path in enumerate(image_paths):
    image = Image.open(img_path)
    image_array = np.array(image)
    
    # Process each variance level of the image 
    for variance in variances:
        # if variance of the noise is 0 keep the original image
        if variance == 0:
            processed_image = Image.fromarray(image_array)
            # For saving: use original image array when variance is 0
            save_array = image_array
        # else add random gaissian noice with the variance to the image 
        else:
            noisy_array = util.random_noise(image_array, mode='gaussian', var=variance)
            noisy_array = (noisy_array * 255).astype(np.uint8)
            processed_image = Image.fromarray(noisy_array)
            # For saving: use noisy array when variance > 0
            save_array = noisy_array
            
        # Save images for the first image only (idx == 0)
        if idx == 0:
            io.imsave(path + f'data/imagenetSubNoise/noisy{idx}_{variance}.jpg', save_array)
        
        # Calculate BRISQUE score for each variance level
        brisque_score = model(processed_image).item()
        results.append({"image_idx" : idx, "variance" : variance,"brisque_score" : brisque_score})

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
plt.savefig(path + 'results/brisque_analysis.png', dpi=800, bbox_inches='tight')
plt.close()
