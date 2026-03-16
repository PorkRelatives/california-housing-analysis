
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("california_housing.csv")

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df["AveOccup"], df["MedHouseVal"], alpha=0.5)

# Add titles and labels
plt.title("Average Occupancy vs. Median House Value")
plt.xlabel("Average Occupancy")
plt.ylabel("Median House Value")
plt.grid(True)

# Save the plot
plt.savefig("occupancy_value_scatterplot.png")

print("Scatter plot saved as occupancy_value_scatterplot.png")
