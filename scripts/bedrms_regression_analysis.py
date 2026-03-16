import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
housing = pd.read_csv("california_housing.csv")

# Filter the data
housing_filtered = housing[housing["AveBedrms"] < 5]

# Prepare the data for regression
X = housing_filtered[["AveBedrms"]]
y = housing_filtered["MedHouseVal"]

# Create and fit the linear regression model
reg = LinearRegression()
reg.fit(X, y)

# Get the R-squared value
r_squared = reg.score(X, y)

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.1, label="Data")

# Plot the regression line
plt.plot(X, reg.predict(X), color='red', linewidth=2, label="Regression Line")

# Add labels and title
plt.xlabel("Average Number of Bedrooms (AveBedrms)")
plt.ylabel("Median House Value (MedHouseVal)")
plt.title("Relationship between Average Bedrooms and Median House Value (AveBedrms < 5)")

# Add the R-squared value to the plot
plt.text(0.05, 0.95, f'R-squared = {r_squared:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

# Add a legend
plt.legend()

# Save the figure
plt.savefig("bedrms_regression_analysis.png")

print("Regression analysis plot saved as bedrms_regression_analysis.png")
