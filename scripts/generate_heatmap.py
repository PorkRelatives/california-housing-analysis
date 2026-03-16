
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
housing = pd.read_csv("california_housing.csv")

# Create the heatmap plot
plt.figure(figsize=(10, 7))
plt.scatter(housing["Longitude"], housing["Latitude"], c=housing["MedHouseVal"], 
            alpha=0.4, s=housing["Population"]/100, label="Population",
            cmap=plt.get_cmap("jet"))

# Add labels and title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("California Housing Prices Heatmap")

# Add a colorbar
cbar = plt.colorbar()
cbar.set_label("Median House Value")



# Add a legend for the size of the dots
plt.legend()

# Save the figure
plt.savefig("house_price_heatmap.png")

print("Heatmap saved as house_price_heatmap.png")
