
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'california_housing.csv'
housing_df = pd.read_csv(file_path)

# --- Preprocessing ---
# Feature Engineering
housing_df['bed_to_room_ratio'] = housing_df['AveBedrms'] / housing_df['AveRooms']
housing_df.replace([np.inf, -np.inf], np.nan, inplace=True)
housing_df.dropna(inplace=True)

# Data Cleaning
housing_df = housing_df[housing_df['AveOccup'] < 10]
housing_df = housing_df[housing_df['AveRooms'] < 40]

# --- Feature Selection ---
location_features = ['Latitude', 'Longitude']
adjustment_features = ['MedInc', 'AveOccup', 'bed_to_room_ratio']
all_features = location_features + adjustment_features

X = housing_df[all_features]
y = housing_df['MedHouseVal']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Stage 1: Location Model ---
print("Training Stage 1: Location Model...")
location_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
location_model.fit(X_train[location_features], y_train)

# --- Calculate Residuals ---
print("Calculating residuals...")
location_predictions_train = location_model.predict(X_train[location_features])
residuals_train = y_train - location_predictions_train

# --- Stage 2: Residual Model ---
print("Training Stage 2: Residual Model...")
residual_model = Ridge(alpha=1.0)
residual_model.fit(X_train[adjustment_features], residuals_train)

# --- Final Prediction on Test Set ---
print("Making final predictions on the test set...")
# Stage 1 prediction
location_predictions_test = location_model.predict(X_test[location_features])
# Stage 2 prediction
residual_predictions_test = residual_model.predict(X_test[adjustment_features])

# Final prediction
final_predictions = location_predictions_test + residual_predictions_test

# --- Evaluation ---
rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
print(f"Two-Stage Model Root Mean Squared Error: {rmse}")

# For comparison, let's train a single model with all features
print("\nTraining a single Gradient Boosting model for comparison...")
single_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
single_model.fit(X_train, y_train)
single_model_predictions = single_model.predict(X_test)
single_model_rmse = np.sqrt(mean_squared_error(y_test, single_model_predictions))
print(f"Single Model Root Mean Squared Error: {single_model_rmse}")
