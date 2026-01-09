import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('results/ks_test/ks_test_results_1000_[0.0, 0.0001]_res_32_180.csv')

# Create figure with subplots
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot mean_i (BRISQUE) on primary y-axis
color1 = 'tab:blue'
ax1.set_xlabel('Resolution', fontsize=12)
ax1.set_ylabel('BRISQUE Mean (mean_i)', color=color1, fontsize=12)
ax1.plot(df['resolution'], df['mean_i'], 'o-', color=color1, label='BRISQUE Mean', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

# Create second y-axis for SSIM
ax2 = ax1.twinx()
color2 = 'tab:green'
ax2.set_ylabel('SSIM Mean', color=color2, fontsize=12)
ax2.plot(df['resolution'], df['ssim_mean'], 's-', color=color2, label='SSIM Mean', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)

# Create third y-axis for PSNR
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
color3 = 'tab:red'
ax3.set_ylabel('PSNR Mean', color=color3, fontsize=12)
ax3.plot(df['resolution'], df['psnr_mean'], '^-', color=color3, label='PSNR Mean', linewidth=2)
ax3.tick_params(axis='y', labelcolor=color3)

# Add title
plt.title('Image Quality Metrics vs Resolution', fontsize=14, pad=20)

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='best')

plt.tight_layout()
plt.savefig('metrics_vs_resolution.png', dpi=300, bbox_inches='tight')
plt.show()
