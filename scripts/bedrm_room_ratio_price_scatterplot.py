import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
housing = pd.read_csv("california_housing.csv")

# Calculate the ratio of average bedrooms to average rooms
housing["Bedrms_ratio"] = housing["AveBedrms"] / housing["AveRooms"]

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(housing["Bedrms_ratio"], housing["MedHouseVal"], alpha=0.1)

# Add labels and title
plt.xlabel("Average Bedrooms to Rooms Ratio")
plt.ylabel("Median House Value (MedHouseVal)")
plt.title("Relationship between Bedrooms/Rooms Ratio and Median House Value")

# Save the figure
plt.savefig("bedrm_room_ratio_price_scatterplot.png")

print("Scatter plot saved as bedrm_room_ratio_price_scatterplot.png")
