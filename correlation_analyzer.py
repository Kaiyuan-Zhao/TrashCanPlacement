import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load heatmap data
path_data = np.loadtxt('path_coverage_matrix.txt')
trash_data = np.loadtxt('trash_collection_matrix.txt')

# Flatten arrays for correlation calculation
path_flat = path_data.flatten()
trash_flat = trash_data.flatten()

# Calculate Pearson correlation
corr, _ = pearsonr(path_flat, trash_flat)
print(f"Pearson correlation coefficient: {corr:.3f}")

# Create figure with 2 subplots
fig = plt.figure(figsize=(16, 8))

# Combined heatmap (simple average)
ax1 = fig.add_subplot(121)
combined = (path_data/np.max(path_data)) + (trash_data/np.max(trash_data))
im1 = ax1.imshow(combined, cmap='hot', interpolation='nearest',
                vmin=0, vmax=2)  # Scale from 0 to 2 since we're adding two normalized maps
ax1.set_title('Combined Heatmap (Sum of Normalized Values)')
plt.colorbar(im1, ax=ax1, label='Sum (0-2 scale)\n0=No activity\n2=Max in both datasets')

# Correlation heatmap (element-wise product)
ax2 = fig.add_subplot(122)
corr_heatmap = (path_data/np.max(path_data)) * (trash_data/np.max(trash_data))
im2 = ax2.imshow(corr_heatmap, cmap='hot', interpolation='nearest',
                vmin=0, vmax=1)  # Scale from 0 to 1 since we're multiplying two normalized maps
ax2.set_title('Correlation Heatmap (Product of Normalized Values)')
plt.colorbar(im2, ax=ax2, label='Product (0-1 scale)\n0=No correlation\n1=Perfect correlation')

# Print correlation coefficient
print(f"Pearson correlation between path coverage and trash collection: {corr:.3f}")

plt.tight_layout()

plt.show()
