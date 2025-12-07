from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from classes.helperfunctions import *



imTrue = fix_dimension(orig_img)

imRecon = fix_dimension(recon)

ssim_val = structural_similarity(imTrue, imRecon, channel_axis=-1, data_range=1.0)
psnr_val = peak_signal_noise_ratio(imTrue, imRecon, data_range=1.0)

print("SSIM:", ssim_val)
print("PSNR:", psnr_val)
