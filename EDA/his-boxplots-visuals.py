import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Directory containing cleaned CSVs
cleaned_dir = "../CleanedDataset"

# 2. Collect all cleaned files
file_paths = glob.glob(os.path.join(cleaned_dir, "*.csv"))
if not file_paths:
    raise FileNotFoundError(f"No CSVs found in {cleaned_dir}")

# 3. Numeric columns to summarize
num_cols = ["temperature", "humidity", "irradiance", "potential", "wind_speed"]

# 4. Combine all towns for distribution plots
combined = pd.concat(
    [pd.read_csv(p).assign(town=os.path.splitext(os.path.basename(p))[0])
     for p in file_paths],
    ignore_index=True
)

# 5. Plot distributions
for var in num_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(
        data=combined,
        x=var,
        hue="town",
        stat="density",
        common_norm=False,
        kde=True,
        element="step",
        alpha=0.5
    )
    plt.title(f"Distribution of {var.capitalize()} by Town")
    plt.xlabel(var.capitalize())
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()