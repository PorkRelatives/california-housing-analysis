
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
housing = pd.read_csv("california_housing.csv")

# Define boundaries for the San Francisco Bay Area
lat_min, lat_max = 37.1, 38.5
lon_min, lon_max = -123, -121.5

# Filter the data for the SF Bay Area
sf_housing = housing[
    (housing['Latitude'] >= lat_min) & (housing['Latitude'] <= lat_max) &
    (housing['Longitude'] >= lon_min) & (housing['Longitude'] <= lon_max)
]

# Create a figure with high resolution
plt.figure(figsize=(12, 9), dpi=300)

# Create the scatter plot
plt.scatter(sf_housing["Longitude"], sf_housing["Latitude"], c=sf_housing["MedHouseVal"], 
            alpha=0.4, s=5, cmap=plt.get_cmap("jet"))

# Add labels and title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("San Francisco Bay Area Housing Prices")

# Add a colorbar
cbar = plt.colorbar()
cbar.set_label("Median House Value")

plt.axis('equal')

# Save the figure
plt.savefig("sf_bay_area_heatmap.png")

print("San Francisco Bay Area heatmap saved as sf_bay_area_heatmap.png")
