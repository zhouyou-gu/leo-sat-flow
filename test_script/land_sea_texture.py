import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
from PIL import Image


# Create a small figure that will serve as your texture image.
fig = plt.figure(figsize=(4, 4), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Set global extent and remove axis for a clean look.
ax.set_global()
ax.set_axis_off()

# Add simple features:
ax.add_feature(cfeature.OCEAN, facecolor='white')
ax.add_feature(cfeature.LAND, facecolor='gray')

# latitudes = np.arange(-90, 91, 30)
# longitudes = np.arange(-180, 181, 30)

# # Draw gridlines without labels (for use as a 3D ball texture)
# ax.gridlines(draw_labels=False, xlocs=longitudes, ylocs=latitudes, 
#              color='black', linestyle='--', linewidth=0.25)

# gl.xformatter = LongitudeFormatter()
# gl.yformatter = LatitudeFormatter()

# Save the figure to a PNG file with no margins.

temp_filename = 'land_sea_texture.png'
plt.savefig(temp_filename, dpi=2000, bbox_inches='tight', pad_inches=0)
plt.close(fig)

# --- Step 2: Force the saved image to be strictly black and white ---
# Open the image using Pillow.
img = Image.open(temp_filename)

# Convert the image to 1-bit pixels (black and white)
bw_img = img.convert('1')

# Save the final black-and-white image.
final_filename = 'land_sea_texture_bw.png'
bw_img.save(final_filename)

print("Texture image saved as land_sea_texture.png")