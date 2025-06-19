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

# 2. Derive town names from filenames
towns = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]

# 3. Variables to plot
vars_to_plot = ["temperature", "humidity", "irradiance", "potential", "wind_speed"]

# 4. Create output directory
output_dir = "seasonal_subseries_plot"
os.makedirs(output_dir, exist_ok=True)

# 5. Seaborn style
sns.set(style="whitegrid")

# 6. Loop over each town
for town, path in zip(towns, file_paths):
    # Load data
    df = pd.read_csv(path, parse_dates=["date"])
    # Extract month number
    df["month"] = df["date"].dt.month

    # Determine subplot grid
    n = len(vars_to_plot)
    cols = 3
    rows = math.ceil(n / cols)

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)

    # Plot each variable
    for idx, var in enumerate(vars_to_plot):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        sns.boxplot(
            data=df,
            x="month",
            y=var,
            ax=ax,
            color="tab:blue"
        )
        ax.set_title(var.replace("_", " ").title(), fontsize=14)
        ax.set_xlabel("")  # month labels on bottom row only
        ax.set_ylabel(var.replace("_", " ").title(), fontsize=12)
        ax.grid(axis="y", alpha=0.75)
        # show month numbers
        ax.set_xticklabels(range(1,13))

    # Remove any empty subplots
    for idx in range(n, rows*cols):
        r, c = divmod(idx, cols)
        fig.delaxes(axes[r][c])

    # Overall title and layout
    fig.suptitle(f"{town} – Monthly Boxplots", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure
    out_path = os.path.join(output_dir, f"{town}_monthly_boxplots.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

print(f"Saved monthly boxplots for all towns in '{output_dir}'")
