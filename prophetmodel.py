import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error # Import mean_squared_error instead
import numpy as np
import matplotlib.pyplot as plt # For plotting

# Load data
df = pd.read_csv("../../DataPreprocessingAndFeatureEngineering/OutlierHandledDataset/Bafoussam_treated.csv")
# df = pd.read_csv("../../CleanedDataset/Bafoussam_IrrPT.csv")


# Convert 'date' to datetime and rename columns for Prophet
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
df = df.rename(columns={'date': 'ds', 'irradiance': 'y'})

# Drop rows where 'ds' became NaT due to coerce errors (if any)
df.dropna(subset=['ds'], inplace=True)

# Ensure data is sorted by date for time series splitting
df = df.sort_values(by='ds')

# Define additional regressors (features that influence irradiance)
# Ensure these columns exist in your DataFrame
regressors = ['temperature', 'humidity', 'potential', 'wind_speed']

# --- Time Series Split (Crucial for proper evaluation) ---
# Option 1: Split by a specific date (recommended for production-like scenarios)
# split_date = pd.to_datetime('YYYY-MM-DD') # Replace with an actual date in your dataset
# train_df = df[df['ds'] <= split_date]
# test_df = df[df['ds'] > split_date]

# Option 2: Split by a percentage of the *latest* data for testing (common in analysis)
# Let's use 80% for training, 20% for testing
split_point_index = int(len(df) * 0.8)
train_df = df.iloc[:split_point_index]
test_df = df.iloc[split_point_index:]

print(f"Training data range: {train_df['ds'].min()} to {train_df['ds'].max()}")
print(f"Testing data range: {test_df['ds'].min()} to {test_df['ds'].max()}")
print(f"Number of training samples: {len(train_df)}")
print(f"Number of testing samples: {len(test_df)}")


# Initialize Prophet model
# Consider weekly_seasonality if your data spans multiple weeks
# Set daily_seasonality=False if you only have one observation per day
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False) # Changed daily_seasonality

# Add additional regressors
for col in regressors:
    if col in df.columns: # Check if column exists to prevent errors
        model.add_regressor(col)
    else:
        print(f"Warning: Regressor '{col}' not found in DataFrame. Skipping.")

# Fit the model
model.fit(train_df)

# Create future DataFrame
# This future DataFrame must also contain the values for the regressors for the prediction period
future = model.make_future_dataframe(periods=len(test_df), freq='D')

# Now, crucially, merge the regressor values from your original 'test_df' into 'future'
# Prophet needs these values for prediction.
# Make sure 'future' only covers the dates you intend to forecast.
future = future.merge(df[regressors + ['ds']], on='ds', how='left')

# Check for NaNs in regressors in 'future' (important!)
if future[regressors].isnull().any().any():
    print("Warning: NaNs found in regressor columns in the future DataFrame. This will cause errors in prediction.")
    print("Please ensure your 'future' DataFrame has complete regressor data for the forecast period.")
    # You might need to fill NaNs, or ensure your merge covers all future dates with data.
    # For this specific case, since we are merging from df, NaNs shouldn't occur unless dates are missing.

# Generate forecast
forecast = model.predict(future)

# Merge predictions and actuals for evaluation
# Ensure we only merge on the dates that are common to both `forecast` and `test_df`
# This usually means taking the forecast for the test period.
forecast_result = forecast[['ds', 'yhat']].merge(test_df[['ds', 'y']], on='ds', how='inner')

# Ensure data types are correct
forecast_result = forecast_result.astype({'y': float, 'yhat': float})

# Calculate metrics
y_true = forecast_result['y']
y_pred = forecast_result['yhat']

# Calculate RMSE manually or use mean_squared_error
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse) # Calculate RMSE from MSE
mae = mean_absolute_error(y_true, y_pred)

# MAPE calculation: Handle division by zero or very small actual values
# Add a small epsilon to y_true to avoid division by zero
epsilon = 1e-8
mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# Plotting
fig1 = model.plot(forecast)
plt.title('Prophet Forecast with Actuals')
plt.xlabel('Date')
plt.ylabel('Irradiance')
plt.show()

fig2 = model.plot_components(forecast)
plt.show()