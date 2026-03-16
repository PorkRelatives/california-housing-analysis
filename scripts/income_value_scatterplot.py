
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('california_housing.csv')

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['MedInc'], df['MedHouseVal'], alpha=0.5)

# Add titles and labels
plt.title('Median Income vs. Median House Value')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.grid(True)

# Save the plot
plt.savefig('income_value_scatterplot.png')

print("Scatter plot saved as income_value_scatterplot.png")
