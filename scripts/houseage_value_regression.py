
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("california_housing.csv")

# Create the regression plot
plt.figure(figsize=(10, 6))
sns.regplot(x="HouseAge", y="MedHouseVal", data=df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})

# Add titles and labels
plt.title("House Age vs. Median House Value with Regression Line")
plt.xlabel("House Age")
plt.ylabel("Median House Value")
plt.grid(True)

# Save the plot
plt.savefig("houseage_value_regression.png")

print("Regression plot saved as houseage_value_regression.png")
