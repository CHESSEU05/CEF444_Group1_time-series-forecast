import os
import glob
import pandas as pd

# 1. Point to your cleaned data
cleaned_dir = "../CleanedDataset"
file_paths = glob.glob(os.path.join(cleaned_dir, "*.csv"))

# 2. Variables for which to detect outliers
vars_to_check = ["temperature", "humidity", "irradiance", "potential", "wind_speed"]

# 3. Collect per-town, per-variable outlier stats
outlier_stats = []

for path in file_paths:
    town = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path, parse_dates=["date"])
    
    for var in vars_to_check:
        series = df[var].dropna()
        
        # compute IQR bounds
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # identify outliers
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        count = outliers.count()
        total = series.count()
        
        outlier_stats.append({
            "town": town,
            "variable": var,
            "outlier_count": count,
            "total_obs": total,
        })

# 4. Build a summary table
outlier_df = pd.DataFrame(outlier_stats)

# 5. Display it
print("\n=== Outlier Summary (1.5×IQR rule) ===")
print(outlier_df.pivot(index="variable", columns="town", 
                       values=["outlier_count"]))
