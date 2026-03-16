from sklearn.datasets import fetch_california_housing
import pandas as pd

# Fetch the dataset
housing = fetch_california_housing()

# Create a pandas DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# Save the DataFrame to a CSV file
df.to_csv('california_housing.csv', index=False)

print("Dataset downloaded and saved to california_housing.csv")