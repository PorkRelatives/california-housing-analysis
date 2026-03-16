
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
housing = pd.read_csv("california_housing.csv")

# Create a figure with higher resolution
plt.figure(figsize=(12, 9), dpi=600)

# Create the scatter plot with all data points
plt.scatter(housing["Longitude"], housing["Latitude"], c=housing["MedHouseVal"], 
            alpha=0.4, s=1, cmap=plt.get_cmap("jet"))

# Add labels and title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("California Housing Prices High-Resolution Heatmap")

# Add a colorbar
cbar = plt.colorbar()
cbar.set_label("Median House Value")

# Save the figure
plt.savefig("high_res_heatmap.png")

print("High-resolution heatmap saved as high_res_heatmap.png")
