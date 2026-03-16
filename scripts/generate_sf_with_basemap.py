
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

# Load the dataset
housing = pd.read_csv("california_housing.csv")

# Define boundaries for the San Francisco Bay Area
lat_min, lat_max = 37.1, 38.5
lon_min, lon_max = -123, -121.5

housing = housing[
    (housing['Latitude'] >= lat_min) & (housing['Latitude'] <= lat_max) &
    (housing['Longitude'] >= lon_min) & (housing['Longitude'] <= lon_max)
]

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    housing, 
    geometry=gpd.points_from_xy(housing.Longitude, housing.Latitude),
    crs="EPSG:4326"
)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 12))

# Plot the housing data
gdf.plot(ax=ax, column='MedHouseVal', cmap='viridis', 
         alpha=0.6, legend=True, markersize=15)

# Add the basemap
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

# Set title
ax.set_title('San Francisco Bay Area Housing Prices with Basemap')

# Save the figure
plt.savefig("sf_bay_area_with_basemap.png", dpi=300)

print("SF Bay Area map with basemap saved as sf_bay_area_with_basemap.png")
