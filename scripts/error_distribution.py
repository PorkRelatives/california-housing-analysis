
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the dataset
file_path = 'california_housing.csv'
housing_df = pd.read_csv(file_path)

# Preprocessing
housing_df['bed_to_room_ratio'] = housing_df['AveBedrms'] / housing_df['AveRooms']
housing_df.replace([np.inf, -np.inf], np.nan, inplace=True)
housing_df.dropna(inplace=True)
housing_df = housing_df[housing_df['AveOccup'] < 10]
housing_df = housing_df[housing_df['AveRooms'] < 40]

# Feature Selection
location_features = ['Latitude', 'Longitude']
adjustment_features = ['MedInc', 'AveOccup', 'bed_to_room_ratio']
all_features = location_features + adjustment_features
X = housing_df[all_features]
y = housing_df['MedHouseVal']

# --- Define Models ---

# 1. Two-Stage Model
location_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
residual_model = Ridge(alpha=1.0)

# 2. Gradient Boosting Model
gbr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('gbr', GradientBoostingRegressor(learning_rate=0.1, max_depth=5, n_estimators=200, random_state=42))
])

# 3. Linear Model (ElasticNet -> Lasso)
linear_model = Pipeline([
    ('scaler', StandardScaler()),
    ('linear', ElasticNet(alpha=0.001, l1_ratio=1.0, max_iter=10000, random_state=42))
])

# --- Cross-validation and Prediction ---

# Two-Stage Model Predictions
location_predictions = cross_val_predict(location_model, X[location_features], y, cv=5)
residuals = y - location_predictions
residual_predictions = cross_val_predict(residual_model, X[adjustment_features], residuals, cv=5)
two_stage_predictions = location_predictions + residual_predictions

# Gradient Boosting Predictions
gbr_predictions = cross_val_predict(gbr_model, X, y, cv=5)

# Linear Model Predictions
linear_predictions = cross_val_predict(linear_model, X, y, cv=5)

# --- Calculate Errors ---
errors_two_stage = y - two_stage_predictions
errors_gbr = y - gbr_predictions
errors_linear = y - linear_predictions

# --- Plot Error Distributions ---
plt.figure(figsize=(12, 8))
sns.kdeplot(errors_two_stage, label='Two-Stage Model', color='red', fill=True)
sns.kdeplot(errors_gbr, label='Gradient Boosting Model', color='blue', fill=True)
sns.kdeplot(errors_linear, label='Linear Model', color='green', fill=True)

plt.title('Error Distribution of Prediction Models')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.savefig('error_distribution.png')
plt.show()

print("Error distribution plot saved as error_distribution.png")
