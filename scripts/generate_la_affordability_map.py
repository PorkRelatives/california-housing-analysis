
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

# Load the dataset
housing = pd.read_csv("california_housing.csv")

# Calculate the price-to-income ratio
housing["PriceToIncomeRatio"] = housing["MedHouseVal"] / housing["MedInc"]

# Define boundaries for the Los Angeles Metropolitan Area
lat_min, lat_max = 33.5, 34.5
lon_min, lon_max = -119, -117

housing_la = housing[
    (housing['Latitude'] >= lat_min) & (housing['Latitude'] <= lat_max) &
    (housing['Longitude'] >= lon_min) & (housing['Longitude'] <= lon_max)
]

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    housing_la, 
    geometry=gpd.points_from_xy(housing_la.Longitude, housing_la.Latitude),
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
ax.set_title('Los Angeles Metro Area Housing Affordability (Price-to-Income Ratio)')

# Save the figure
plt.savefig("la_metro_area_affordability_map.png", dpi=300)

print("LA Metro Area affordability map saved as la_metro_area_affordability_map.png")
