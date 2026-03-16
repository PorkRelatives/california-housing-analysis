
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('california_housing.csv')
df['affordability'] = df['MedHouseVal'] / df['MedInc']

# Create and save the histogram
plt.figure(figsize=(10, 6))
plt.hist(df['affordability'], bins=50, edgecolor='black')
plt.title('Distribution of Affordability (House Price to Income Ratio)')
plt.xlabel('Affordability (House Price / Income)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('affordability_distribution.png')
print("Affordability distribution plot saved to affordability_distribution.png")

# Create and save the density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(df['affordability'], fill=True)
plt.title('Density Plot of Affordability (House Price to Income Ratio)')
plt.xlabel('Affordability (House Price / Income)')
plt.ylabel('Density')
plt.grid(True)
plt.savefig('affordability_density.png')
print("Affordability density plot saved to affordability_density.png")
