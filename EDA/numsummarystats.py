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

# 4. Build a list of DataFrames for descriptive stats
stats_frames = []
for path in file_paths:
    town = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path, parse_dates=["date"])
    
    desc = df[num_cols].describe().T.rename(columns={"50%": "median"})
    desc["town"] = town
    desc.reset_index(inplace=True)
    desc.rename(columns={"index": "variable"}, inplace=True)
    stats_frames.append(desc)

# 5. Concatenate and display summary table
summary_df = pd.concat(stats_frames, ignore_index=True)
print("\n=== Numerical Summary by Town ===")
print(summary_df[["town","variable","mean","median","std","min","25%","75%","max"]])

# 6. Combine all towns for distribution plots
combined = pd.concat(
    [pd.read_csv(p).assign(town=os.path.splitext(os.path.basename(p))[0])
     for p in file_paths],
    ignore_index=True
)

# 7. Plot distributions
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

# 8. Categorical frequencies (if any)
for path in file_paths:
    town = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path)
    cat_cols = df.select_dtypes(include=["object","category"]).columns
    if cat_cols.any():
        print(f"\n=== Categorical Frequencies for {town} ===")
        for col in cat_cols:
            counts = df[col].value_counts(dropna=False)
            print(f"\n{col}:\n{counts}")
            plt.figure(figsize=(6, 3))
            sns.countplot(data=df, x=col, order=counts.index)
            plt.title(f"{col} Frequencies in {town}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
