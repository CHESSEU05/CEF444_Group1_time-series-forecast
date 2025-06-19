import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Load cleaned data
cleaned_dir = "../../CleanedDataset"
files = glob.glob(os.path.join(cleaned_dir, "*.csv"))

# 2. Variables for analysis
vars_to_decompose = ["irradiance", "temperature", "humidity"]

# 3. Output directory for decomposition plots
decompose_dir = "seasonal_decompose_plots"
os.makedirs(decompose_dir, exist_ok=True)

# 4. Loop through each town
for file in files:
    town = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file, parse_dates=["date"])
    df.set_index("date", inplace=True)

    # Resample data to hourly averages for clearer decomposition (if needed)
    df_hourly = df.resample('H').mean()

    for var in vars_to_decompose:
        # Ensure no missing values (interpolation)
        series = df_hourly[var].interpolate(method='time')

        # Decompose with a daily frequency (24 hours)
        result = seasonal_decompose(series, model='additive', period=24)

        # Plot decomposition results
        fig = result.plot()
        fig.set_size_inches(12, 8)
        for ax in fig.axes:
            ax.grid(axis='both', linestyle='--', alpha=0.7)
        fig.suptitle(f'{town} - Seasonal Decomposition of {var.title()}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save plot
        fig.savefig(os.path.join(decompose_dir, f"{town}_{var}_decompose.png"), dpi=300)
        plt.close(fig)

# # 5. Seasonal subseries plots for irradiance
# subseries_dir = "seasonal_subseries_plots"
# os.makedirs(subseries_dir, exist_ok=True)

# for file in files:
#     town = os.path.splitext(os.path.basename(file))[0]
#     df = pd.read_csv(file, parse_dates=["date"])
#     df.set_index("date", inplace=True)

#     # Extract hour and month for grouping
#     df['hour'] = df.index.hour
#     df['month'] = df.index.month

#     # Pivot table: mean irradiance per hour per month
#     pivot = df.pivot_table(values='irradiance', index='hour', columns='month', aggfunc='mean')

#     # Plot subseries
#     plt.figure(figsize=(12, 6))
#     pivot.plot(ax=plt.gca())
#     plt.title(f'{town} - Seasonal Subseries of Irradiance (Hourly by Month)', fontsize=16)
#     plt.xlabel('Hour of the Day')
#     plt.ylabel('Average Irradiance (W/m²)')
#     plt.grid(True, alpha=0.5)
#     plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()

#     # Save plot
#     plt.savefig(os.path.join(subseries_dir, f"{town}_irradiance_subseries.png"), dpi=300)
#     plt.close()

print(f"Seasonal decomposition plots saved in '{decompose_dir}'")
# print(f"Seasonal subseries plots saved in '{subseries_dir}'")
