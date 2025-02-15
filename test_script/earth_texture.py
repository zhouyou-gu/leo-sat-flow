import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np

# Create a small figure that will serve as your texture image.
fig = plt.figure(figsize=(4, 4), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Set global extent and remove axis for a clean look.
ax.set_global()
ax.set_axis_off()

# Add simple features:
ax.add_feature(cfeature.OCEAN, facecolor='deepskyblue')
ax.add_feature(cfeature.LAND, facecolor='white')
ax.add_feature(cfeature.RIVERS, linewidth=1)
ax.add_feature(cfeature.LAKES, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

latitudes = np.arange(-90, 91, 30)
longitudes = np.arange(-180, 181, 30)

# Draw gridlines without labels (for use as a 3D ball texture)
ax.gridlines(draw_labels=False, xlocs=longitudes, ylocs=latitudes, 
             color='black', linestyle='--', linewidth=0.25)

# gl.xformatter = LongitudeFormatter()
# gl.yformatter = LatitudeFormatter()

# Save the figure to a PNG file with no margins.
plt.savefig('simple_earth_texture.png', dpi=300 ,bbox_inches='tight', pad_inches=0)
plt.close(fig)

print("Texture image saved as simple_earth_texture.png")