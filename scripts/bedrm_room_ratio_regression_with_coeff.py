import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# Load the dataset
housing = pd.read_csv("california_housing.csv")

# Calculate the ratio of average bedrooms to average rooms
housing["Bedrms_ratio"] = housing["AveBedrms"] / housing["AveRooms"]

# Filter the data
filtered_housing = housing[housing["Bedrms_ratio"] < 0.4]

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(filtered_housing["Bedrms_ratio"], filtered_housing["MedHouseVal"], alpha=0.1)

# Add a regression line
m, b = np.polyfit(filtered_housing["Bedrms_ratio"], filtered_housing["MedHouseVal"], 1)
plt.plot(filtered_housing["Bedrms_ratio"], m*filtered_housing["Bedrms_ratio"] + b, color='red')

# Calculate R-squared
r2 = r2_score(filtered_housing["MedHouseVal"], m*filtered_housing["Bedrms_ratio"] + b)

# Add labels and title
plt.xlabel("Average Bedrooms to Rooms Ratio")
plt.ylabel("Median House Value (MedHouseVal)")
plt.title("Relationship between Bedrooms/Rooms Ratio (<0.4) and Median House Value with Regression Line")

# Add regression coefficient and R-squared to the plot
plt.text(0.25, 4, f'Regression Coefficient: {m:.2f}\nR-squared: {r2:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Save the figure
plt.savefig("bedrm_room_ratio_regression_with_coeff.png")

print("Scatter plot with regression line and coefficients saved as bedrm_room_ratio_regression_with_coeff.png")
