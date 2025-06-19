import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Directory containing cleaned CSVs
cleaned_dir = "../../CleanedDataset"
file_paths = glob.glob(os.path.join(cleaned_dir, "*.csv"))
if not file_paths:
    raise FileNotFoundError(f"No CSVs found in {cleaned_dir}")

# 2. Variables to plot
vars_to_plot = ["temperature", "humidity", "irradiance", "potential", "wind_speed"]

# 3. Seaborn style
sns.set(style="whitegrid")

# 4. Output directory for time-series plots
output_dir = "time_series_plots"
os.makedirs(output_dir, exist_ok=True)

# 5. Loop through each town
for path in file_paths:
    town = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    
    # A. Full span line plots (one subplot per variable)
    fig, axes = plt.subplots(len(vars_to_plot), 1, figsize=(12, 3 * len(vars_to_plot)), sharex=True)
    for ax, var in zip(axes, vars_to_plot):
        sns.lineplot(data=df, x=df.index, y=var, ax=ax)
        ax.set_title(f"{town}: {var.replace('_',' ').title()} Over Time")
        ax.set_ylabel(var.replace('_',' ').title())
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{town}_full_timeseries.png"), dpi=300)
    plt.close(fig)
    
    # B. Zoomed-in – last 7 days
    df_week = df.last("7D")
    fig, axes = plt.subplots(len(vars_to_plot), 1, figsize=(12, 3 * len(vars_to_plot)), sharex=True)
    for ax, var in zip(axes, vars_to_plot):
        sns.lineplot(data=df_week, x=df_week.index, y=var, ax=ax)
        ax.set_title(f"{town}: {var.replace('_',' ').title()} – Last 7 Days")
        ax.set_ylabel(var.replace('_',' ').title())
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{town}_last7d_timeseries.png"), dpi=300)
    plt.close(fig)
    
    # C. Zoomed-in – last 30 days
    df_month = df.last("30D")
    fig, axes = plt.subplots(len(vars_to_plot), 1, figsize=(12, 3 * len(vars_to_plot)), sharex=True)
    for ax, var in zip(axes, vars_to_plot):
        sns.lineplot(data=df_month, x=df_month.index, y=var, ax=ax)
        ax.set_title(f"{town}: {var.replace('_',' ').title()} – Last 30 Days")
        ax.set_ylabel(var.replace('_',' ').title())
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{town}_last30d_timeseries.png"), dpi=300)
    plt.close(fig)

    # D. January 2024
    df_jan = df.loc['2024-01-01':'2024-01-31']
    fig, axes = plt.subplots(len(vars_to_plot), 1, figsize=(12, 3 * len(vars_to_plot)), sharex=True)
    for ax, var in zip(axes, vars_to_plot):
        sns.lineplot(data=df_jan, x=df_jan.index, y=var, ax=ax)
        ax.set_title(f"{town}: {var.replace('_',' ').title()} – January 2024")
        ax.set_ylabel(var.replace('_',' ').title())
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{town}_jan2024_timeseries.png"), dpi=300)
    plt.close(fig)

    # E. Last 7 days of January 2024
    df_jan7 = df.loc['2024-01-25':'2024-01-31']
    fig, axes = plt.subplots(len(vars_to_plot), 1, figsize=(12, 3 * len(vars_to_plot)), sharex=True)
    for ax, var in zip(axes, vars_to_plot):
        sns.lineplot(data=df_jan7, x=df_jan7.index, y=var, ax=ax)
        ax.set_title(f"{town}: {var.replace('_',' ').title()} – Jan 25–31, 2024")
        ax.set_ylabel(var.replace('_',' ').title())
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{town}_jan25_31_2024_timeseries.png"), dpi=300)
    plt.close(fig)

print(f"Time series plots (full span, last 7d, last 30d) saved in '{output_dir}'")
