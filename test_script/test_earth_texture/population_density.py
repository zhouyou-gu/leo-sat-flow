from matplotlib.colors import LinearSegmentedColormap
import rasterio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from rasterio.plot import plotting_extent

tif_file = rasterio.open('GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif')
ghs_data = tif_file.read()
import numpy as np

print("Tiff Boundary", tif_file.bounds)
print("Tiff CRS", tif_file.crs)
print("Data shape", ghs_data.shape, "Ratio", ghs_data.shape[2] / ghs_data.shape[1])
print("Max value", np.amax(ghs_data))
print("Min value", np.amin(ghs_data))
print("Sum value", np.sum(ghs_data))

ghs_data[0][ghs_data[0] < 0.0] = 0.0

ghs_data_pic = ghs_data[0]  # Use the first band for population density
ghs_data_pic = np.log1p(ghs_data_pic*100)  # Apply log transformation to enhance visibility
ghs_data_pic[ghs_data_pic < 0] = 0  # Ensure no negative values after log transformation
ghs_data_pic = ghs_data_pic / np.max(ghs_data_pic)  # Normalize the data to [0, 1]
# Create a figure with Cartopy to visualize the data
fig = plt.figure(figsize=(8, 8), dpi=500)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_global()
ax.set_axis_off()
ax.add_feature(cfeature.OCEAN, facecolor='white')
ax.add_feature(cfeature.LAND, facecolor='#A0A0A0')  # Light gray for land
ax.add_feature(cfeature.COASTLINE, edgecolor='#606060', linewidth=0.1)
# Create a colormap for the texture, where the low values are transparent
colors = [(0, 0.5, 1, a) for a in np.linspace(0, 1, 256)]
test_cmap = LinearSegmentedColormap.from_list("test_cmap", colors, N=256)

# set the extent of the plot to match the raster data
ax.set_extent([tif_file.bounds.left, tif_file.bounds.right, tif_file.bounds.bottom, tif_file.bounds.top], crs=ccrs.PlateCarree())
# Plot the population density data, filling the entire extent of the map
ax.imshow(ghs_data_pic, extent=plotting_extent(tif_file), transform=ccrs.PlateCarree(), cmap=test_cmap, vmin=0, vmax=1)
# Merge the texture with the land and sea features
plt.savefig('population_density_texture.png', dpi=500, bbox_inches='tight', pad_inches=0, transparent=True)


from skimage.measure import block_reduce
A = 20
ghs_data = block_reduce(ghs_data[0], (A, A), func=np.sum)  # Downsample by a factor of 20
# Export the data as a numpy array
print("ghs_data shape", ghs_data.shape)
print("ghs_data max", np.amax(ghs_data))
print("ghs_data min", np.amin(ghs_data))
print("ghs_data sum", np.sum(ghs_data))
# save the reduced data as a numpy array
np.save('population_density_texture.npy', ghs_data)