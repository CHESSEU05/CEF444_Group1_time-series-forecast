import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set(style="whitegrid")
TARGET_YEAR = 1950

# Define Cameroon seasonal boundaries
SEASON_LINES = {
    'Start Rainy': f'{TARGET_YEAR}-05-01',
    'End Rainy': f'{TARGET_YEAR}-10-31'
}

# File paths per city
files = {
    'Bafoussam': '../CleanedDataset/Bafoussam_IrrPT.csv',
    'Bambili': '../CleanedDataset/Bambili_IrrPT.csv',
    'Bamenda': '../CleanedDataset/Bamenda_IrrPT.csv',
    'Yaounde': '../CleanedDataset/Yaounde_IrrPT.csv'
}

for city, path in files.items():
    print(f"\n📍 Processing: {city}")

    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        continue

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    date_col = next((col for col in df.columns if 'date' in col), None)
    irr_col = next((col for col in df.columns if 'irr' in col), None)

    if not date_col or not irr_col:
        print(f"⚠️ Missing columns for {city}.")
        continue

    df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d', errors='coerce')
    df[irr_col] = pd.to_numeric(df[irr_col], errors='coerce')
    df = df.dropna(subset=[date_col, irr_col])
    df = df.set_index(date_col)
    df = df[df.index.year == TARGET_YEAR]

    if df.empty:
        print(f"⚠️ No data for year {TARGET_YEAR} in {city}.")
        continue

    output_dir = f'plots/{city}_{TARGET_YEAR}_enhanced'
    os.makedirs(output_dir, exist_ok=True)

    # Compute rolling and monthly averages
    df['rolling_irr'] = df[irr_col].rolling(window=30, min_periods=1).mean()
    df['month'] = df.index.month
    monthly_avg = df.groupby('month')[irr_col].mean()

    # Create the enhanced plot
    plt.figure(figsize=(16, 6))
    sns.lineplot(x=df.index, y=df[irr_col], label='Daily Irradiance', color='skyblue')
    sns.lineplot(x=df.index, y=df['rolling_irr'], label='30-day Rolling Avg', color='green')

    # Monthly average overlay
    for month in monthly_avg.index:
        month_days = df[df.index.month == month]
        if not month_days.empty:
            avg_val = monthly_avg[month]
            plt.hlines(y=avg_val, xmin=month_days.index.min(), xmax=month_days.index.max(),
                       color='orange', linestyle='--', linewidth=1.2,
                       label='Monthly Avg' if month == 1 else "")

    # Add vertical lines for season boundaries
    for label, date_str in SEASON_LINES.items():
        plt.axvline(pd.to_datetime(date_str), color='red', linestyle=':', linewidth=1.5, label=label)

    # Labels and decorations
    plt.title(f"{city} - Daily Irradiance with Seasonal Context ({TARGET_YEAR})")
    plt.xlabel("Date")
    plt.ylabel("Irradiance")
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save plot
    plt.savefig(f'{output_dir}/{city}_irradiance_enhanced.png')
    plt.close()

    print(f"✅ Enhanced plot saved for {city}: {output_dir}/{city}_irradiance_enhanced.png")
