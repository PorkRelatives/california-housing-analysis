
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
housing = pd.read_csv("california_housing.csv")

# Define boundaries for the Los Angeles Metropolitan Area
lat_min, lat_max = 33.5, 34.5
lon_min, lon_max = -119, -117

# Filter the data for the LA Metro Area
la_housing = housing[
    (housing['Latitude'] >= lat_min) & (housing['Latitude'] <= lat_max) &
    (housing['Longitude'] >= lon_min) & (housing['Longitude'] <= lon_max)
]

# Create a figure with high resolution
plt.figure(figsize=(12, 9), dpi=300)

# Create the scatter plot
plt.scatter(la_housing["Longitude"], la_housing["Latitude"], c=la_housing["MedHouseVal"], 
            alpha=0.4, s=5, cmap=plt.get_cmap("jet"))

# Add labels and title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Los Angeles Metropolitan Area Housing Prices")

# Add a colorbar
cbar = plt.colorbar()
cbar.set_label("Median House Value")

plt.axis('equal')

# Save the figure
plt.savefig("la_metro_area_heatmap.png")

print("Los Angeles Metropolitan Area heatmap saved as la_metro_area_heatmap.png")
