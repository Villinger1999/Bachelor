import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pyiqa
import zipfile
from PIL import Image
from skimage import util, io
import numpy as np
import sys
import scipy.stats as stats
import random as random



np.random.seed(42)  # Set a fixed seed for reproducibility
random.seed(42)

#setup brique model
model = pyiqa.create_metric('brisque', device="cpu")

results = []

ks_results = []

# function to get the paths to "image_count" images in a folder at "image_dir" 
def get_images(image_dir, image_count):
    all_files = [f for f in os.listdir(image_dir) if f.endswith('.JPEG')]
    all_files.sort()  # Ensures deterministic order
    selected_files = all_files[:image_count]
    return [os.path.join(image_dir, fname) for fname in selected_files]
   
if __name__ == "__main__":
    
    ##------ setup of the variable needed to do test the brisque scores 
 
    # Make path the folder path
    path = os.getcwd() + "/" # always points to the folder you are in
    defaults = [
        "/dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/test/",  # image_paths
        0.01,    # variance
        32,      # res_lb
        96,      # res_ub
        4,       # res_step
        100,     # image_count
        "no_plot"  # plots_download
    ]

    # Fill in sys.argv with defaults if not enough arguments are provided
    args = sys.argv[1:] + [None] * (7 - len(sys.argv[1:]))
    args = [a if a not in [None, "None", "def"] else d for a, d in zip(args, defaults)]

    image_paths = args[0]
    var_arr = [0.0, float(args[1])]
    res_lb = int(args[2])
    res_ub = int(args[3])
    res_step = int(args[4])
    image_count = int(args[5])
    plots_download = args[6].lower() == "plot"
    
    # creates the list of resolutions that are to be compared
    resolution_arr = list(range(res_lb, res_ub+1,res_step))    
    
    # getting the first "image_count" images from the folder containing the data data
    image_paths = get_images(image_paths, image_count)
    
    for reso in resolution_arr:
        # Process images - back to your original simple approach
        for idx, img_path in enumerate(image_paths):
            image = Image.open(img_path)
            image = image.resize((reso, reso))  # Resize to resolution x resolution
            image_array = np.array(image)
            
            # Process each variance level of the image 
            for variance in var_arr:
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
                    io.imsave(path + f'data/imagenetSubNoise/noisy{idx}_{variance}_{reso}x{reso}.jpg', save_array)
                
                # Calculate BRISQUE score for each variance level
                brisque_score = model(processed_image).item()
                
                ##---- save outliers uncomment the line below to save all the images that has a value lower than 0 and higher than 100
                # if brisque_score < 0 or brisque_score > 100:
                #     io.imsave(path + f'data/invalid_brisque/noisy{idx}_{variance}_{reso}x{reso}_brisque_{brisque_score:.2f}.jpg', save_array)
                
                results.append({"resolution" : reso, "image_idx" : idx, "variance" : variance, "brisque_score" : brisque_score})

        df_results = pd.DataFrame(results, columns=["resolution", "image_idx", "variance", "brisque_score"])
        
        # Extract just the BRISQUE scores for the histogram
        brisque_scores = df_results['brisque_score'].tolist()
        # print(brisque_scores)


        ##------- Plots and save the histograms of the two distribution, at each resolution 
        if plots_download == True:
            fig, ax1 = plt.subplots(figsize=(8, 5))
            max_count = 0
            for variance in var_arr:
                var_scores = df_results[(df_results['resolution'] == reso) & (df_results['variance'] == variance)]['brisque_score']
                # Plot histogram (frequency, not density)
                counts, bins, patches = ax1.hist(var_scores, bins=20, alpha=0.6, label=f'var={variance}', density=False)
                max_count = max(max_count, counts.max() if len(counts) > 0 else 0)
            ax1.set_xlabel('BRISQUE Score')
            ax1.set_ylabel('Frequency (Count)')
            ax1.set_ylim(0, image_count)
            # Plot CDF on secondary y-axis
            ax2 = ax1.twinx()
            for variance in var_arr:
                var_scores = df_results[(df_results['resolution'] == reso) & (df_results['variance'] == variance)]['brisque_score']
                sorted_scores = np.sort(var_scores)
                cdf = np.arange(1, len(sorted_scores)+1) / len(sorted_scores) if len(sorted_scores) > 0 else []
                ax2.plot(sorted_scores, cdf, marker='.', linestyle='-', label=f'CDF var={variance}')
            ax2.set_ylabel('CDF')
            ax2.set_ylim(0, 1)
            # Combine legends from both axes
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(handles1 + handles2, labels1 + labels2, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=len(var_arr)*2)
            plt.title(f'BRISQUE Scores by Noise Variance (Var1 = {var_arr[0]}, Var2 = {var_arr[1]},{reso}x{reso})')
            ax1.grid(True, alpha=0.3)
            # Ensure the results folder exists before saving
            results_folder = os.path.join(path, 'results')
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
            plt.savefig(path + f'results/brisque_analysis_{var_arr}_{reso}x{reso}.png', dpi=800, bbox_inches='tight')
            plt.close()
        
        #------- make the ks_test on the two variances chosen
        brisque_var_0 = df_results[(df_results["resolution"] == reso) & (df_results['variance'] == var_arr[0])]['brisque_score']
        brisque_var_i = df_results[(df_results["resolution"] == reso) & (df_results['variance'] == var_arr[1])]['brisque_score']
        ks_test = stats.ks_2samp(brisque_var_0, brisque_var_i, alternative='two-sided', mode='asymp')
        ks_results.append({"resolution": reso, "ks_statistic": np.round(ks_test.statistic,2), "p_value": ks_test.pvalue, "statistic_location": np.round(ks_test.statistic_location,2)})

        df_ks = pd.DataFrame(ks_results)
        df_ks.to_csv(path + f'ks_test_results_{var_arr}_res_{resolution_arr[0]}_{resolution_arr[-1]}.csv', index=False)