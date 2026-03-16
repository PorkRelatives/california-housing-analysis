
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'california_housing.csv'
housing_df = pd.read_csv(file_path)

# Feature Engineering
housing_df['bed_to_room_ratio'] = housing_df['AveBedrms'] / housing_df['AveRooms']

# Replace infinite values with NaN
housing_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values that we might have created
housing_df.dropna(inplace=True)

# Data Cleaning: Handle outliers based on the report
housing_df = housing_df[housing_df['AveOccup'] < 10]
housing_df = housing_df[housing_df['AveRooms'] < 40]


# Feature Selection
features = ['Latitude', 'Longitude', 'MedInc', 'AveOccup', 'bed_to_room_ratio']
X = housing_df[features]
y = housing_df['MedHouseVal']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Regressor model
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gbr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gbr.predict(X_test)

# Calculate and print the Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")

