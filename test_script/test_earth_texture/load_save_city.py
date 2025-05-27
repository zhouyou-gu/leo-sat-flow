import numpy as np
import pandas as pd

# Load the CSV file (adjust the file name/path as needed)
df = pd.read_csv('worldcities.csv')

# Drop rows missing key columns (lat, lng, or population)
df = df.dropna(subset=['lat', 'lng', 'population'])

# Define latitude and longitude bin edges (10-degree bins)
lat_bins = np.arange(-90, 91, 10)   # from -90 to 90
lng_bins = np.arange(-180, 181, 10)  # from -180 to 180

# Create new columns that represent the bin each city falls into
df['lat_bin'] = pd.cut(df['lat'], bins=lat_bins)
df['lng_bin'] = pd.cut(df['lng'], bins=lng_bins)

# Set the number of cities you want per grid cell
K = 10  # Adjust this number as needed

# Define a function to sample up to K rows from each bin group
def sample_group(group):
    return group.sample(n=min(K, len(group)), random_state=42)

# Group by the latitude and longitude bins and apply sampling
balanced_df = df.groupby(['lat_bin', 'lng_bin'], group_keys=False).apply(sample_group)

balanced_df = balanced_df.sample(n=100, random_state=42)

# Select only the columns we want: lat, lng, and population
result = balanced_df[['lat', 'lng', 'population']]

# Save the resulting DataFrame to a new CSV file without header and index
result.to_csv('cities.csv', index=False, header=False)

print("Balanced cities data saved to balanced_cities.csv")