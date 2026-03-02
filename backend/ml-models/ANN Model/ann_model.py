import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# This makes the script always look for files in the same folder as the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load datasets
corn = pd.read_csv(os.path.join(BASE_DIR, 'IntegratedCorn_dataset.csv'))
oats = pd.read_csv(os.path.join(BASE_DIR, 'IntegratedOats_dataset.csv'))
soy  = pd.read_csv(os.path.join(BASE_DIR, 'IntegratedSoybean_dataset.csv'))
soil = pd.read_csv(os.path.join(BASE_DIR, 'ohio_soil_data.csv'), sep=';')

print("Corn shape:", corn.shape)
print("Oats shape:", oats.shape)
print("Soybean shape:", soy.shape)
print("Soil shape:", soil.shape)

# Add a crop type label to each dataset
corn['crop'] = 'corn'
oats['crop'] = 'oats'
soy['crop']  = 'soybean'

# Rename soybean columns to match corn and oats column names
soy = soy.rename(columns={
    'PRCP_total' : 'precip_total',
    'PRCP_avg'   : 'precip_avg',
    'PRCP_max'   : 'precip_max',
    'TAVG_avg'   : 'Temp_avg',
    'TMAX_avg'   : 'TempMAX_avg',
    'TMIN_avg'   : 'TempMIN_avg',
    'SNOW_total' : 'Snow_total',
    'SNWD_avg'   : 'SnowDepth_avg',
    'AWND_avg'   : 'WindSpeed_avg',
})

print("Crop labels added and soybean columns renamed!")
print("Soy columns sample:", list(soy.columns[:10]))



# Define the columns we want to keep from all three datasets
common_cols = [
    'year', 'county', 'crop', 'stage', 'yield_bu_acre',
    'acres_planted', 'acres_harvested',
    'opt_temp_min', 'opt_temp_max', 'opt_precip',
    'stage_duration_days', 'temp_deviation', 'precip_deviation',
    'water_stress_indicator', 'total_extreme_weather_days',
    'hail_events', 'tornado_events', 'high_wind_events', 'thunder_events', 'GDD',
    'precip_total', 'precip_avg', 'precip_max',
    'Temp_avg', 'TempMAX_avg', 'TempMIN_avg',
    'Snow_total', 'SnowDepth_avg', 'WindSpeed_avg'
]

# Select only the available common columns from each dataset
# (some datasets may not have every column)
corn_clean = corn[[c for c in common_cols if c in corn.columns]]
oats_clean = oats[[c for c in common_cols if c in oats.columns]]
soy_clean  = soy[[c for c in common_cols if c in soy.columns]]

# Stack all three into one combined dataset
combined = pd.concat([corn_clean, oats_clean, soy_clean], ignore_index=True)

print("Combined shape:", combined.shape)
print("Crops in dataset:", combined['crop'].value_counts().to_string())

# Identify columns to aggregate (everything except the grouping columns)
agg_cols = [c for c in combined.columns if c not in ['year', 'county', 'crop', 'stage']]

# Average all numeric columns across stages
# Result: one row per county/year/crop
aggregated = combined.groupby(['year', 'county', 'crop'])[agg_cols].mean().reset_index()

print("Aggregated shape:", aggregated.shape)
print("Sample:")
print(aggregated.head(3).to_string(index=False))


# Merge soil data into the aggregated dataset by matching county and year
merged = pd.merge(aggregated, soil, on=['county', 'year'], how='left')

print("After soil merge:", merged.shape)

# Fill missing values using the median for each crop type
# This way corn missing values are filled with corn medians
numeric_cols = merged.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if merged[col].isnull().sum() > 0:
        merged[col] = merged.groupby('crop')[col].transform(
            lambda x: x.fillna(x.median())
        )

print("Missing values after cleaning:", merged.isnull().sum().sum())
print("Final dataset shape:", merged.shape)

# Encode crop as a number since the ANN can't read text
# corn=0, oats=1, soybean=2
merged['crop_encoded'] = merged['crop'].map({'corn': 0, 'oats': 1, 'soybean': 2})

# Define the input features (what the model learns from)
feature_cols = [
    'crop_encoded', 'year',
    'opt_temp_min', 'opt_temp_max', 'opt_precip',
    'temp_deviation', 'precip_deviation', 'water_stress_indicator',
    'total_extreme_weather_days', 'GDD',
    'precip_total', 'Temp_avg', 'TempMAX_avg', 'TempMIN_avg',
    'avg_ph', 'avg_organic_matter_pct', 'avg_water_capacity',
    'avg_bulk_density', 'avg_clay_pct', 'avg_sand_pct', 'avg_silt_pct',
    'avg_cation_exchange_capacity', 'avg_saturated_hydraulic_conductivity',
    'avg_soil_temperature_0_to_7cm', 'avg_soil_moisture_0_to_7cm'
]

# X = inputs, y = what we are trying to predict (yield)
X = merged[feature_cols]
y = merged['yield_bu_acre']

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Feature columns:", feature_cols)

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Normalize features so all values are on the same scale
# The ANN performs much better when features are normalized
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on training data only
X_test_scaled  = scaler.transform(X_test)        # apply same scale to test data

print("Features normalized!")
print("Sample scaled values (first row):", X_train_scaled[0].round(3))

# Build the ANN model
# hidden_layer_sizes=(128, 64, 32) means 3 layers with 128, 64, and 32 neurons
# relu = activation function that helps the model learn complex patterns
# adam = optimizer that adjusts the model weights during training
# early_stopping = stops training if the model stops improving
ann = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)

# Train the model
print("Training the ANN model...")
ann.fit(X_train_scaled, y_train)
print("Training complete!")

# Make predictions on the test set
y_pred = ann.predict(X_test_scaled)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n=== ANN Model Results ===")
print(f"R² Score:  {r2:.4f}  (1.0 = perfect, 0.0 = no better than guessing)")
print(f"RMSE:      {rmse:.2f} bushels/acre")
print(f"MAE:       {mae:.2f} bushels/acre")

# Show sample predictions vs actual
print("\nSample Predictions vs Actual:")
results = pd.DataFrame({
    'Crop'     : merged.loc[y_test.index, 'crop'].values[:10],
    'County'   : merged.loc[y_test.index, 'county'].values[:10],
    'Actual'   : y_test.values[:10],
    'Predicted': y_pred[:10].round(1)
})
print(results.to_string(index=False))




# Save the trained model
with open(os.path.join(BASE_DIR, 'ann_model.pkl'), 'wb') as f:
    pickle.dump(ann, f)

# Save the scaler (needed to normalize new data before predicting)
with open(os.path.join(BASE_DIR, 'ann_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# Save the clean merged dataset
merged.to_csv(os.path.join(BASE_DIR, 'ann_training_dataset.csv'), index=False)

print("\nFiles saved:")
print("  - ann_model.pkl")
print("  - ann_scaler.pkl")
print("  - ann_training_dataset.csv")