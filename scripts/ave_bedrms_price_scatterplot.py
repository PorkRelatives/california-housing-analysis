import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
housing = pd.read_csv("california_housing.csv")

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(housing["AveBedrms"], housing["MedHouseVal"], alpha=0.1)

# Add labels and title
plt.xlabel("Average Number of Bedrooms (AveBedrms)")
plt.ylabel("Median House Value (MedHouseVal)")
plt.title("Relationship between Average Bedrooms and Median House Value")

# Save the figure
plt.savefig("ave_bedrms_price_scatterplot.png")

print("Scatter plot saved as ave_bedrms_price_scatterplot.png")
