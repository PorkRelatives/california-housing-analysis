
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Load the dataset
df = pd.read_csv("california_housing.csv")

# Filter out extreme outliers in AveOccup for better visualization and regression fitting
df_filtered = df[df['AveOccup'] < 50]

# Perform linear regression on the filtered data
slope, intercept, r_value, p_value, std_err = linregress(df_filtered["AveOccup"], df_filtered["MedHouseVal"])

# Create the regression plot
plt.figure(figsize=(12, 7))
sns.regplot(x="AveOccup", y="MedHouseVal", data=df_filtered, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})

# Add titles and labels
plt.title("Average Occupancy vs. Median House Value with Regression Line")
plt.xlabel("Average Occupancy")
plt.ylabel("Median House Value")
plt.grid(True)

# Add the regression equation and R-squared value to the plot
text_str = f'k (slope) = {slope:.2f}\nr (correlation) = {r_value:.2f}'
plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

# Save the plot
plt.savefig("occupancy_price_regression.png")

print("Regression plot with stats saved as occupancy_price_regression.png")
