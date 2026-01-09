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
import scipy.stats as stats
import random as random
import argparse
import skimage.metrics



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
    parser = argparse.ArgumentParser(description="BRISQUE score analysis")
    parser.add_argument('--image_paths', type=str, default="/dtu/datasets1/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/test/", help='Path to image directory')
    parser.add_argument('--variance', type=float, default=0.01, help='Variance for noise')
    parser.add_argument('--res_lb', type=int, default=32, help='Lower bound for resolution')
    parser.add_argument('--res_ub', type=int, default=96, help='Upper bound for resolution')
    parser.add_argument('--res_step', type=int, default=4, help='Step size for resolution')
    parser.add_argument('--image_count', type=int, default=100, help='Number of images to use')
    parser.add_argument('--plot', action='store_true', default=False, help='Enable plotting (default: False)')
    args = parser.parse_args()
    image_paths = args.image_paths
    var_arr = [0.0, args.variance]
    res_lb = args.res_lb
    res_ub = args.res_ub
    res_step = args.res_step
    image_count = args.image_count
    plots_download = args.plot
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
                    noisy_array = image_array  # For SSIM/PSNR reference
                # else add random gaissian noice with the variance to the image 
                else:
                    noisy_array = util.random_noise(image_array, mode='gaussian', var=variance)
                    noisy_array = (noisy_array * 255).astype(np.uint8)
                    processed_image = Image.fromarray(noisy_array)
                    # For saving: use noisy array when variance > 0
                    save_array = noisy_array
                # Save images for the first image only (idx == 0)
                if idx == 3:
                    io.imsave(path + f'data/imagenetSubNoise/noisy{idx}_{variance}_{reso}x{reso}.jpg', save_array)
                # Calculate BRISQUE score for each variance level
                brisque_score = model(processed_image).item()
                if idx == 0:
                    print(f"BRISQUE score of the first image with noise variance {variance} is {brisque_score}")
                # Calculate SSIM and PSNR between original and noisy image (only if variance > 0, else set to nan or perfect)
                if variance == 0:
                    ssim_score = 1.0
                    psnr_score = float('inf')
                else:
                    # Ensure both arrays are in the same shape and type
                    try:
                        ssim_score = skimage.metrics.structural_similarity(image_array, noisy_array, data_range=255, channel_axis=-1)
                    except TypeError:
                        # For older skimage
                        ssim_score = skimage.metrics.structural_similarity(image_array, noisy_array, data_range=255, multichannel=True)
                    psnr_score = skimage.metrics.peak_signal_noise_ratio(image_array, noisy_array, data_range=255)
                results.append({"resolution" : reso, "image_idx" : idx, "variance" : variance, "brisque_score" : brisque_score, "ssim": ssim_score, "psnr": psnr_score})

        df_results = pd.DataFrame(results, columns=["resolution", "image_idx", "variance", "brisque_score", "ssim", "psnr"])
        
        


        ##------- Plots and save the histograms of the two distribution, at each resolution 
        if plots_download == True:
            fig, ax1 = plt.subplots(figsize=(8, 5))
            max_count = 0
            # Gather all brisque scores for this resolution
            all_brisque_scores = df_results[df_results['resolution'] == reso]['brisque_score']
            if len(all_brisque_scores) > 0:
                min_brisque = all_brisque_scores.min()
                max_brisque = all_brisque_scores.max()
                if min_brisque >= 0:
                    x_min = 0
                else:
                    x_min = int(np.floor(min_brisque / 10.0)) * 10
                x_max = int(np.ceil(max_brisque / 10.0)) * 10
            else:
                x_min = 0
                x_max = 100
            for variance in var_arr:
                var_scores = df_results[(df_results['resolution'] == reso) & (df_results['variance'] == variance)]['brisque_score']
                # Plot histogram (frequency, not density)
                counts, bins, patches = ax1.hist(var_scores, bins=20, alpha=0.6, label=f'var={variance}', density=False, range=(x_min, x_max))
                max_count = max(max_count, counts.max() if len(counts) > 0 else 0)
            ax1.set_xlabel('BRISQUE Score')
            ax1.set_ylabel('Frequency (Count)')
            ax1.set_ylim(0, image_count)
            ax1.set_xlim(x_min, x_max)
            # Plot CDF on secondary y-axis
            ax2 = ax1.twinx()
            for variance in var_arr:
                var_scores = df_results[(df_results['resolution'] == reso) & (df_results['variance'] == variance)]['brisque_score']
                sorted_scores = np.sort(var_scores)
                cdf = np.arange(1, len(sorted_scores)+1) / len(sorted_scores) if len(sorted_scores) > 0 else []
                ax2.plot(sorted_scores, cdf, marker='.', linestyle='-', label=f'CDF var={variance}')
            ax2.set_ylabel('CDF')
            ax2.set_ylim(0, 1)
            ax2.set_xlim(x_min, x_max)
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
            plt.savefig(path + f'results/brisque_analysis_{image_count}_{var_arr}_{reso}x{reso}.png', dpi=800, bbox_inches='tight')
            plt.close()
        
        #------- make the ks_test on the two variances chosen
        brisque_var_0 = df_results[(df_results["resolution"] == reso) & (df_results['variance'] == var_arr[0])]['brisque_score']
        brisque_var_i = df_results[(df_results["resolution"] == reso) & (df_results['variance'] == var_arr[1])]['brisque_score']
        ks_test = stats.ks_2samp(brisque_var_0, brisque_var_i, alternative='two-sided', mode='asymp')

        #------ FPR and FNR calculation
        # Use the KS statistic location as threshold
        threshold = ks_test.statistic_location
        fpr = np.sum(brisque_var_0 > threshold) / len(brisque_var_0) if len(brisque_var_0) > 0 else np.nan
        fnr = np.sum(brisque_var_i < threshold) / len(brisque_var_i) if len(brisque_var_i) > 0 else np.nan

        # Mean and variance for each distribution
        mean_0 = np.round(np.mean(brisque_var_0),2) if len(brisque_var_0) > 0 else np.nan
        var_0 = np.round(np.var(brisque_var_0),2) if len(brisque_var_0) > 0 else np.nan
        mean_i = np.round(np.mean(brisque_var_i),2) if len(brisque_var_i) > 0 else np.nan
        var_i = np.round(np.var(brisque_var_i),2) if len(brisque_var_i) > 0 else np.nan

        # KS test against normal distribution with same mean/var
        if len(brisque_var_0) > 1:
            normal_0 = np.random.normal(mean_0, np.sqrt(var_0), size=len(brisque_var_0))
            ks_norm_0 = stats.ks_2samp(brisque_var_0, normal_0, alternative='two-sided', mode='asymp')
            ks_norm_0_stat = ks_norm_0.statistic
            ks_norm_0_p = ks_norm_0.pvalue
        else:
            ks_norm_0_stat = np.nan
            ks_norm_0_p = np.nan
        if len(brisque_var_i) > 1:
            normal_i = np.random.normal(mean_i, np.sqrt(var_i), size=len(brisque_var_i))
            ks_norm_i = stats.ks_2samp(brisque_var_i, normal_i, alternative='two-sided', mode='asymp')
            ks_norm_i_stat = ks_norm_i.statistic
            ks_norm_i_p = ks_norm_i.pvalue
        else:
            ks_norm_i_stat = np.nan
            ks_norm_i_p = np.nan

        # Calculate SSIM and PSNR statistics for var_arr[1] (non-zero variance)
        ssim_vals = df_results[(df_results['resolution'] == reso) & (df_results['variance'] == var_arr[1])]['brisque']
        psnr_vals = df_results[(df_results['resolution'] == reso) & (df_results['variance'] == var_arr[1])]['psnr']
        ssim_mean = np.round(np.mean(ssim_vals), 4) if len(ssim_vals) > 0 else np.nan
        ssim_var = np.round(np.var(ssim_vals), 4) if len(ssim_vals) > 0 else np.nan
        psnr_mean = np.round(np.mean(psnr_vals), 2) if len(psnr_vals) > 0 else np.nan
        psnr_var = np.round(np.var(psnr_vals), 2) if len(psnr_vals) > 0 else np.nan

        ks_results.append({
            "resolution": reso,
            "ks_statistic": np.round(ks_test.statistic,2),
            "p_value": np.round(ks_test.pvalue,4),
            "statistic_location": np.round(threshold,2),
            "FPR": np.round(fpr,3),
            "FNR": np.round(fnr,3),
            "mean_0": np.round(mean_0,2),
            "var_0": np.round(var_0,2),
            "ks_norm_0_stat": np.round(ks_norm_0_stat,3),
            "ks_norm_0_p": np.round(ks_norm_0_p,4),
            "mean_i": np.round(mean_i,2),
            "var_i": np.round(var_i,2),
            "ks_norm_i_stat": np.round(ks_norm_i_stat,3),
            "ks_norm_i_p": np.round(ks_norm_i_p,4),
            "ssim_mean": ssim_mean,
            "ssim_var": ssim_var,
            "psnr_mean": psnr_mean,
            "psnr_var": psnr_var
        })
        
        df_ks = pd.DataFrame(ks_results)
        df_ks.to_csv(path + f'ks_test_results_{image_count}_{var_arr}_res_{resolution_arr[0]}_{resolution_arr[-1]}.csv', index=False)