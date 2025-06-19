import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# 1. Point to cleaned data folder
cleaned_dir = "../CleanedDataset"
files = glob.glob(os.path.join(cleaned_dir, "*.csv"))

# 2. Variables of interest
vars_to_check = ["temperature", "humidity", "irradiance", "potential", "wind_speed"]

for path in files:
    town = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path, parse_dates=["date"])
    
    # 3. Compute missing counts & percentages
    miss_count = df[vars_to_check].isna().sum()
    miss_pct   = df[vars_to_check].isna().mean() * 100
    summary = pd.DataFrame({
        "missing_count": miss_count,
        "missing_pct":   miss_pct.round(2)
    })
    print(f"\n=== Missing Data for {town} ===")
    print(summary)
