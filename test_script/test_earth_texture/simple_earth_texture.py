from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np


# Create a figure with Cartopy to visualize the data
fig = plt.figure(figsize=(8, 8), dpi=500)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_global()
ax.set_axis_off()
ax.add_feature(cfeature.OCEAN, facecolor='white')
ax.add_feature(cfeature.LAND, facecolor="#D4D4D4")  # Light gray for land
ax.add_feature(cfeature.COASTLINE, edgecolor='#606060', linewidth=0.1)
# Create a colormap for the texture, where the low values are transparent
colors = [(0, 0.8, 0.25, a) for a in np.linspace(0, 1, 256)]
test_cmap = LinearSegmentedColormap.from_list("test_cmap", colors, N=256)

plt.savefig('simple_earth_texture.png', dpi=500, bbox_inches='tight', pad_inches=0, transparent=True)

