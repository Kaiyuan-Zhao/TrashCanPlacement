import numpy as np
import matplotlib.pyplot as plt

# Load heatmap data from files
path_coverage = np.loadtxt('path_coverage_matrix.txt')
trash_collection = np.loadtxt('trash_collection_matrix.txt')

# Display settings from main.py
maxPercentage = 60
paddingMultiplyer = 1

# Display final averaged heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Path coverage heatmap (averaged)
im1 = ax1.imshow(path_coverage, cmap='hot', interpolation='nearest',
                norm=plt.Normalize(vmin=0, vmax=np.max(path_coverage)*paddingMultiplyer))
ax1.set_title("Average Path Coverage")
fig.colorbar(im1, ax=ax1, label="Average Coverage")

# Trash collection heatmap (averaged)
im2 = ax2.imshow(trash_collection, cmap='hot', interpolation='nearest', 
                norm=plt.Normalize(vmin=0, vmax=maxPercentage*paddingMultiplyer))
ax2.set_title("Average Trash Collected")
fig.colorbar(im2, ax=ax2, label="Average Percentage")

plt.tight_layout()
plt.show()
