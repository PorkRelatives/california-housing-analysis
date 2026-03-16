
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
# Stage 1: Location Model
location_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
# Stage 2: Residual Model
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
# Stage 1
location_predictions = cross_val_predict(location_model, X[location_features], y, cv=5)
residuals = y - location_predictions
# Stage 2
residual_predictions = cross_val_predict(residual_model, X[adjustment_features], residuals, cv=5)
two_stage_predictions = location_predictions + residual_predictions

# Gradient Boosting Predictions
gbr_predictions = cross_val_predict(gbr_model, X, y, cv=5)

# Linear Model Predictions
linear_predictions = cross_val_predict(linear_model, X, y, cv=5)

# --- Determine Best Model for Each Location ---
errors = pd.DataFrame({
    'two_stage': (y - two_stage_predictions)**2,
    'gradient_boosting': (y - gbr_predictions)**2,
    'linear': (y - linear_predictions)**2
})

housing_df['best_model'] = errors.idxmin(axis=1)
model_map = {'two_stage': 1, 'gradient_boosting': 2, 'linear': 3}
housing_df['best_model_code'] = housing_df['best_model'].map(model_map)

# --- Calculate and Show Best Model Counts ---
best_model_counts = housing_df['best_model'].value_counts()
print("--- Model Dominance Counts ---")
print(best_model_counts)
print("------------------------------")


# --- Create Heatmap ---
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='Longitude',
    y='Latitude',
    hue='best_model',
    palette=['red', 'blue', 'green'],
    data=housing_df,
    s=20,
    alpha=0.5
)
plt.title('Dominance of House Price Prediction Models in California')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Best Performing Model')
plt.savefig('model_dominance_heatmap.png')
plt.show()

print("Heatmap saved as model_dominance_heatmap.png")
