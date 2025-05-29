import pandas as pd
import numpy as np

# 1. Load the full SatNOGS stations CSV
df = pd.read_csv('satnogs_stations.csv')

# 2. Select only the columns you need
loc_df = df[['id', 'name', 'lat', 'lng']].copy()

# (Optional) If you need a NumPy array of coordinates:
coords = loc_df[['lat', 'lng']].to_numpy()
# e.g. latitudes = coords[:, 0], longitudes = coords[:, 1]

# 3. Peek at the first few rows
print(loc_df.head())

#check whether the latitudes and longitudes are in the range of [-90, 90] and [-180, 180] respectively
if not loc_df['lat'].between(-90, 90).all():
    print("Warning: Some latitudes are out of range [-90, 90]")
if not loc_df['lng'].between(-180, 180).all():
    print("Warning: Some longitudes are out of range [-180, 180]")
# 5. Check for duplicates
duplicates = loc_df.duplicated(subset=['lat', 'lng'])
if duplicates.any():
    print("Warning: Found duplicate locations")
    # print number of duplicates
    print(f"Number of duplicate locations: {duplicates.sum()}")
    loc_df = loc_df[~duplicates]  # Remove duplicates
# double check that the duplicates are removed
duplicates_after_removal = loc_df.duplicated(subset=['lat', 'lng'])
if duplicates_after_removal.any():
    print("Warning: Some duplicates still exist after removal")
    
# 6. Check for NaN values
if loc_df.isnull().values.any():
    print("Warning: Found NaN values in the data")
    loc_df = loc_df.dropna()  # Remove rows with NaN values

# 4. Save to a new CSV
loc_df.to_csv('satnogs_locations.csv', index=False)
print("Saved locations to 'satnogs_locations.csv'")

#show the latitudes and longitudes in a 2D earth map using matplotlib and cartopy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Create a 2D map using Cartopy
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

# Plot the station locations
ax.scatter(loc_df['lng'], loc_df['lat'], color='red', s=10, transform=ccrs.PlateCarree())

plt.title("SatNOGS Station Locations")
plt.show()