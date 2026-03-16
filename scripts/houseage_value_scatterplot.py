
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("california_housing.csv")

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df["HouseAge"], df["MedHouseVal"], alpha=0.5)

# Add titles and labels
plt.title("House Age vs. Median House Value")
plt.xlabel("House Age")
plt.ylabel("Median House Value")
plt.grid(True)

# Save the plot
plt.savefig("houseage_value_scatterplot.png")

print("Scatter plot saved as houseage_value_scatterplot.png")
