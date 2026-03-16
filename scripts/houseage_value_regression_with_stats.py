
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Load the dataset
df = pd.read_csv("california_housing.csv")

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(df["HouseAge"], df["MedHouseVal"])

# Create the regression plot
plt.figure(figsize=(12, 7))
sns.regplot(x="HouseAge", y="MedHouseVal", data=df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})

# Add titles and labels
plt.title("House Age vs. Median House Value with Regression Line")
plt.xlabel("House Age")
plt.ylabel("Median House Value")
plt.grid(True)

# Add the regression equation and R-squared value to the plot
text_str = f'k (slope) = {slope:.2f}\nr (correlation) = {r_value:.2f}'
plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

# Save the plot
plt.savefig("houseage_value_regression_with_stats.png")

print("Regression plot with stats saved as houseage_value_regression_with_stats.png")
