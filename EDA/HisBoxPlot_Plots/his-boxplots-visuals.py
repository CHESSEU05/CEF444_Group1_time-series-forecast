import os
import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Directory containing cleaned CSVs
cleaned_dir = "../../CleanedDataset"
file_paths = glob.glob(os.path.join(cleaned_dir, "*.csv"))
if not file_paths:
    raise FileNotFoundError(f"No CSVs found in {cleaned_dir}")

# 2. List of towns based on filenames
towns = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]

# 3. Variables to plot
vars_to_plot = ["temperature", "humidity", "irradiance", "potential", "wind_speed"]

# 4. Create output directory for plots
output_dir = "his-boxplots-plots"
os.makedirs(output_dir, exist_ok=True)

# 5. Seaborn style
sns.set(style="whitegrid")

# 6. Loop through each town
for town in towns:
    # Load data
    path = os.path.join(cleaned_dir, f"{town}.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    
    # Determine subplot grid
    n = len(vars_to_plot)
    cols = 3
    rows = math.ceil(n / cols)
    
    # A. Multi-panel histogram+KDE
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for idx, var in enumerate(vars_to_plot):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        sns.histplot(
            df[var].dropna(),
            bins=30,
            stat="count",
            kde=True,
            ax=ax,
            color="tab:blue",
            edgecolor="w",
            alpha=0.7
        )
        ax.set_title(var.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.75)
    # Remove empty subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        fig.delaxes(axes[r][c])
    fig.suptitle(f"{town} Variable Distributions", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, f"{town}_hist.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # B. Multi-panel boxplots
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for idx, var in enumerate(vars_to_plot):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        sns.boxplot(
            y=df[var].dropna(),
            ax=ax,
            color="tab:orange"
        )
        ax.set_title(var.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel(var.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.75)
    # Remove empty subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        fig.delaxes(axes[r][c])
    fig.suptitle(f"{town} Variable Boxplots", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, f"{town}_box.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

print(f"All histograms and boxplots for each town saved in '{output_dir}'")
