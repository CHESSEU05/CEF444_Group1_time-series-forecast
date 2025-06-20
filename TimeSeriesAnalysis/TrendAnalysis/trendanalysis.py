import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuration
cleaned_dir    = "../../CleanedDataset"
file_paths     = glob.glob(os.path.join(cleaned_dir, "*.csv"))
vars_to_trend  = ["temperature", "humidity", "irradiance", "potential", "wind_speed"]
window_days    = 30   # 30-days rolling window
out_dir        = "trend_analysis_plots"

# 2. Prepare output directory
os.makedirs(out_dir, exist_ok=True)

# 3. Seaborn style
sns.set(style="whitegrid")

# 4. Loop over each town file
for path in file_paths:
    town = os.path.splitext(os.path.basename(path))[0]
    
    # Load and index by date
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    
    # For each variable, compute rolling stats, trend line, and plot
    for var in vars_to_trend:
        series = df[var].dropna()
        
        # Rolling mean & median (require at least half-window points)
        roll_mean = series.rolling(window=window_days, center=True, min_periods=window_days//2).mean()
        roll_med  = series.rolling(window=window_days, center=True, min_periods=window_days//2).median()
        
        # Fit linear trend
        x      = np.arange(len(series))
        mask   = ~series.isna()
        slope, intercept = np.polyfit(x[mask], series.values[mask], 1)
        trend_line = intercept + slope * x
        
        # Plot everything
        plt.figure(figsize=(12, 4))
        plt.plot(series.index, series,        label="Original",        color="gray",  alpha=0.3)
        plt.plot(roll_mean.index, roll_mean,  label=f"{window_days}days Rolling Mean",   color="blue")
        plt.plot(roll_med.index,  roll_med,   label=f"{window_days}days Rolling Median", color="green", linestyle="--")
        plt.plot(series.index,    trend_line, label="Linear Trend",    color="red",   linewidth=2)
        
        plt.title(f"{town}: Trend Analysis for {var.replace('_',' ').title()}", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel(var.replace('_',' ').title())
        plt.legend()
        plt.grid(axis="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        # Save figure
        fname = f"{town}_{var}_trend.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
        plt.close()

print(f"All trend analysis plots saved in '{out_dir}'")
