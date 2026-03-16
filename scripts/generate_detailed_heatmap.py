
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
housing = pd.read_csv("california_housing.csv")

# Create a figure with higher resolution
plt.figure(figsize=(12, 9), dpi=300)

# Create the scatter plot with all data points
plt.scatter(housing["Longitude"], housing["Latitude"], c=housing["MedHouseVal"], 
            alpha=0.4, cmap=plt.get_cmap("jet"))

# Add labels and title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("California Housing Prices Detailed Heatmap")

# Add a colorbar
cbar = plt.colorbar()
cbar.set_label("Median House Value")

# Save the figure
plt.savefig("detailed_house_price_heatmap.png")

print("Detailed heatmap saved as detailed_house_price_heatmap.png")
