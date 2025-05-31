# load npy file on user population grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from rasterio.plot import plotting_extent
import rasterio
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# Load the population density data
ghs_data = np.load('population_density_texture.npy')
# Create a figure with Cartopy to visualize the data
fig = plt.figure(figsize=(2, 2), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_global()
ax.set_axis_off()
# ax.add_feature(cfeature.OCEAN, facecolor='white')
# ax.add_feature(cfeature.LAND, facecolor="#777272")  # Light gray for land
# ax.add_feature(cfeature.COASTLINE, edgecolor='#606060', linewidth=0.1)

# Create a colormap for the texture, where the low values are transparent
colors = [(0, 0.5, 1, a) for a in np.linspace(0, 1, 256)]
test_cmap = LinearSegmentedColormap.from_list("test_cmap", colors, N=256)
# Plot the population density data, filling the entire extent of the map
ax.imshow(ghs_data, extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree(), cmap=test_cmap, vmin=0, vmax=np.max(ghs_data))
# remove all margins
fig.subplots_adjust(left=0, right=1, top=1, bottom=0,
                    wspace=0, hspace=0)

print("ghs_data shape:", ghs_data.shape)
# transparent backgrounds
# fig.patch.set_alpha(0.0)
# ax.patch.set_alpha(0.0)
plt.show()
