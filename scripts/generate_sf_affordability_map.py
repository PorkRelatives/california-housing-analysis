
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

# Load the dataset
housing = pd.read_csv("california_housing.csv")

# Calculate the price-to-income ratio
housing["PriceToIncomeRatio"] = housing["MedHouseVal"] / housing["MedInc"]

# Define boundaries for the San Francisco Bay Area
lat_min, lat_max = 37.1, 38.5
lon_min, lon_max = -123, -121.5

housing_sf = housing[
    (housing['Latitude'] >= lat_min) & (housing['Latitude'] <= lat_max) &
    (housing['Longitude'] >= lon_min) & (housing['Longitude'] <= lon_max)
]

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    housing_sf, 
    geometry=gpd.points_from_xy(housing_sf.Longitude, housing_sf.Latitude),
    crs="EPSG:4326"
)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 12))

# Plot the housing data
gdf.plot(ax=ax, column='PriceToIncomeRatio', cmap='magma', 
         alpha=0.6, legend=True, markersize=15,
         legend_kwds={'label': "Price-to-Income Ratio"})

# Add the basemap
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

# Set title
ax.set_title('San Francisco Bay Area Housing Affordability (Price-to-Income Ratio)')

# Save the figure
plt.savefig("sf_bay_area_affordability_map.png", dpi=300)

print("SF Bay Area affordability map saved as sf_bay_area_affordability_map.png")
