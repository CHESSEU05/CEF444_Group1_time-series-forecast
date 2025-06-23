import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Try importing Prophet, warn if not installed
try:
    from prophet import Prophet
except ImportError:
    raise ImportError("Prophet is not installed. Please run `pip install prophet` and try again.")

# --- Configuration ---
data_dir = '../../DataPreprocessingAndFeatureEngineering/OutlierHandledDataset'
output_dir = 'prophet_forecast_outputs'
os.makedirs(output_dir, exist_ok=True)
summary_txt_path = os.path.join(output_dir, 'forecast_summary.txt')

regressors = ['temperature', 'humidity', 'potential', 'wind_speed']
output_metrics = []
summary_lines = []

# --- Function to Run Prophet on Each Town ---
def run_prophet_on_town(file_path):
    town_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n{'='*40}\nProcessing Town: {town_name}\n{'='*40}")

    try:
        df = pd.read_csv(file_path)

        # Date parsing and renaming
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        df = df.rename(columns={'date': 'ds', 'irradiance': 'y'})
        df.dropna(subset=['ds'], inplace=True)
        df = df.sort_values(by='ds')

        existing_regressors = [col for col in regressors if col in df.columns]
        missing = set(regressors) - set(existing_regressors)
        if missing:
            print(f"Warning: Missing regressors in {town_name}: {missing}")

        # Time split
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        print(f"Training range: {train_df['ds'].min()} to {train_df['ds'].max()}")
        print(f"Testing range : {test_df['ds'].min()} to {test_df['ds'].max()}")
        print(f"Train samples : {len(train_df)}, Test samples: {len(test_df)}")

        # Prophet model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        for reg in existing_regressors:
            model.add_regressor(reg)

        model.fit(train_df[['ds', 'y'] + existing_regressors])

        future = model.make_future_dataframe(periods=len(test_df), freq='D')
        future = future.merge(df[['ds'] + existing_regressors], on='ds', how='left')

        if future[existing_regressors].isnull().any().any():
            print(f"⚠ NaNs in future regressor data for {town_name}. Skipping.")
            return

        forecast = model.predict(future)

        # Evaluation
        forecast_eval = forecast[['ds', 'yhat']].merge(test_df[['ds', 'y']], on='ds', how='inner')
        forecast_eval = forecast_eval.astype({'y': float, 'yhat': float})

        y_true = forecast_eval['y']
        y_pred = forecast_eval['yhat']
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        print(f"📊 RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%")

        output_metrics.append({
            'Town': town_name,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE (%)': mape
        })

        # Save forecast-only plot
        fig = model.plot(forecast)
        plt.title(f"{town_name} - Prophet Forecast")
        plt.xlabel('Date')
        plt.ylabel('Irradiance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{town_name}_forecast_only.png"))
        plt.close(fig)

        fig_components = model.plot_components(forecast)
        fig_components.savefig(os.path.join(output_dir, f"{town_name}_forecast_components.png"))
        plt.close(fig_components)

        # Save summary text
        summary_lines.append(
            f"Town: {town_name}\n"
            f"Train Range: {train_df['ds'].min().date()} to {train_df['ds'].max().date()}\n"
            f"Test Range: {test_df['ds'].min().date()} to {test_df['ds'].max().date()}\n"
            f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%\n"
            f"{'-'*50}\n"
        )

    except Exception as e:
        print(f"❌ Error processing {town_name}: {e}")

# --- Batch Process All CSVs ---
for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_dir, filename)
        run_prophet_on_town(file_path)

# --- Output Results ---
metrics_df = pd.DataFrame(output_metrics)
metrics_df = metrics_df.sort_values(by='RMSE')

# Save summary to text file
with open(summary_txt_path, 'w') as f:
    f.writelines(summary_lines)

# import ace_tools as tools
# tools.display_dataframe_to_user(name="Prophet Forecasting Metrics by Town", dataframe=metrics_df)
