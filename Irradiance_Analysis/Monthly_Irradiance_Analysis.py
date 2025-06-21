import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set(style="whitegrid")
TARGET_YEAR = 1950

# List of cities and file paths
files = {
    'Bafoussam': '../CleanedDataset/Bafoussam_IrrPT.csv',
    'Bambili': '../CleanedDataset/Bambili_IrrPT.csv',
    'Bamenda': '../CleanedDataset/Bamenda_IrrPT.csv',
    'Yaounde': '../CleanedDataset/Yaounde_IrrPT.csv'
}

# Process each city
for city, path in files.items():
    print(f"\n📍 Processing: {city}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()  # Clean column names

    # Detect column names
    date_col = next((col for col in df.columns if 'date' in col), None)
    irr_col = next((col for col in df.columns if 'irr' in col), None)

    if not date_col or not irr_col:
        print(f"⚠️ Skipping {city} — missing required columns.")
        continue

    # 🔧 Fix: Convert integer date format YYYYMMDD to datetime
    df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d', errors='coerce')
    df[irr_col] = pd.to_numeric(df[irr_col], errors='coerce')

    df = df.dropna(subset=[date_col, irr_col])
    df = df.set_index(date_col)
    df = df[df.index.year == TARGET_YEAR]  # Filter year 1950 only

    if df.empty:
        print(f"⚠️ No data for year {TARGET_YEAR} in {city}.")
        continue

    output_dir = f'plots/{city}_{TARGET_YEAR}'
    os.makedirs(output_dir, exist_ok=True)

    # ✅ 1. Daily Plot (Whole year)
    plt.figure(figsize=(14, 5))
    sns.lineplot(data=df, x=df.index, y=irr_col)
    plt.title(f"{city} - Daily Irradiance ({TARGET_YEAR})")
    plt.xlabel("Date")
    plt.ylabel("Irradiance")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{city}_daily_{TARGET_YEAR}.png')
    plt.close()

    # ✅ 2. Weekly Plots (each 7-day window)
    weekly = df.resample('W')
    for i, (week_start, week_df) in enumerate(weekly):
        if len(week_df) < 2:
            continue
        plt.figure(figsize=(10, 3))
        sns.lineplot(data=week_df, x=week_df.index, y=irr_col, marker='o')
        plt.title(f"{city} - Week {i+1} ({week_start.date()})")
        plt.xlabel("Date")
        plt.ylabel("Irradiance")
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{city}_week_{i+1:02d}.png')
        plt.close()

    # ✅ 3. Monthly Plots
    monthly = df.resample('ME')
    for i, (month_start, month_df) in enumerate(monthly):
        if len(month_df) < 2:
            continue
        plt.figure(figsize=(12, 4))
        sns.lineplot(data=month_df, x=month_df.index, y=irr_col, marker='o')
        month_name = month_start.strftime('%B')
        plt.title(f"{city} - {month_name} Irradiance ({TARGET_YEAR})")
        plt.xlabel("Date")
        plt.ylabel("Irradiance")
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{city}_{month_name}_{TARGET_YEAR}.png')
        plt.close()

    print(f"✅ Finished all plots for {city} — saved to: {output_dir}")
