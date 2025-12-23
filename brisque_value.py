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
def get_consistent_images(image_dir, image_count):
    all_files = [f for f in os.listdir(image_dir) if f.endswith('.JPEG') and f.startswith('ILSVRC2012_test_')]
    all_files.sort()  # Ensures deterministic order
    selected_files = all_files[:image_count]
    return [os.path.join(image_dir, fname) for fname in selected_files]
   
if __name__ == "__main__":
    
    ##------ setup of the variable needed to do test the brisque scores 
 
    # Make path the folder path
    path = os.getcwd() + "/" # always points to the folder you are in
    relative_path = sys.argv[1]

    # the variance of the noise, defualt: 0.01
    if sys.argv[2] != None:
        var_arr=[0.0,float(sys.argv[2])]
    else: var_arr = [0.0,0.01]
    
    #resolution lower bound and upper bound, and the step size used from res_lb to res_ub, default: 32, 96, 4, meaning it rescales from 
    if int(sys.argv[3]) != None:
        res_lb = int(sys.argv[3])
    else: res_lb = 32
    
    if int(sys.argv[4]) != None:
        res_ub = int(sys.argv[4])
    else: res_ub = 96
    
    if int(sys.argv[5]) != None:
        int(sys.argv[5])
    else: res_step = 4
    
    # creates the list of resolutions that are to be compared
    resolution_arr = list(range(res_lb, res_ub+1,res_step))
    
    # get how many images is to be used for the brisque score calculations
    
    if int(sys.argv[6]) != None:
        image_count = int(sys.argv[6])
    else: image_count = 100
    
    # Check if 'plot' is passed as a command-line argument, default: False
    if sys.argv[7].lower() == "plot":
        plots_download = True
    else: plots_download = False
    
    if sys.argv[8] == None:
        image_paths = "/dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/test" #for DTU HPC data set
    else: image_paths = sys.argv[8]
    # getting the first "image_count" images from the folder containing the data data
    image_paths = get_consistent_images(image_paths, image_count)
    
    
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
                if brisque_score < 0 or brisque_score > 100:
                    io.imsave(path + f'data/invalid_brisque/noisy{idx}_{variance}_{reso}x{reso}_brisque_{brisque_score:.2f}.jpg', save_array)
                results.append({"resolution" : reso, "image_idx" : idx, "variance" : variance,"brisque_score" : brisque_score})

        df_results = pd.DataFrame(results)

        # Extract just the BRISQUE scores for the histogram
        brisque_scores = df_results['brisque_score'].tolist()
        # print(brisque_scores)

        ##------- Plots and save the histograms of the two distribution, at each resolution 
        if plots_download == True:
            for variance in var_arr:
                var_scores = df_results[(df_results['resolution'] == reso) & (df_results['variance'] == variance)]['brisque_score']
                # Plot histogram
                plt.hist(var_scores, bins=20, alpha=0.6, label=f'var={variance}', density=True)
                # Plot CDF
                sorted_scores = np.sort(var_scores)
                cdf = np.arange(1, len(sorted_scores)+1) / len(sorted_scores)
                plt.plot(sorted_scores, cdf, marker='.', linestyle='-', label=f'CDF var={variance}')
            plt.xlabel('BRISQUE Score')
            plt.ylabel('Density / CDF')
            plt.title(f'BRISQUE Scores by Noise Variance (Var1 = {var_arr[0]}, Var2 = {var_arr[1]},{reso}x{reso})')
            plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=len(var_arr)*2)
            plt.grid(True, alpha=0.3)
            plt.savefig(path + f'results/brisque_analysis_{reso}x{reso}.png', dpi=800, bbox_inches='tight')
            plt.close()
        
        #------- make the ks_test on the two variances chosen
        brisque_var_0 = df_results[(df_results["resolution"] == reso) & (df_results['variance'] == var_arr[0])]['brisque_score']
        brisque_var_i = df_results[(df_results["resolution"] == reso) & (df_results['variance'] == var_arr[1])]['brisque_score']
        ks_test = stats.ks_2samp(brisque_var_0, brisque_var_i, alternative='two-sided', mode='asymp')
        ks_results.append({"resolution": reso, "ks_statistic": np.round(ks_test.statistic,2), "p_value": ks_test.pvalue, "statistic_location": np.round(ks_test.statistic_location,2)})

        df_ks = pd.DataFrame(ks_results)
        df_ks.to_csv(path + f'results/ks_test_results_var{var_arr[0]}_{int(var_arr[1])}_res_{resolution_arr[0]}_{resolution_arr[-1]}.csv', index=False)