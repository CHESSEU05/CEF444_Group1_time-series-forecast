import pandas as pd
import matplotlib.pyplot as plt
import os

# List of datasets
files = {
    'Bafoussam': 'CleanedDataset/Bafoussam_IrrPT.csv',
    'Bambili': 'CleanedDataset/Bambili_IrrPT.csv',
    'Bamenda': 'CleanedDataset/Bamenda_IrrPT.csv',
    'Yaounde': 'CleanedDataset/Yaounde_IrrPT.csv'
}

# Create output directories
os.makedirs('plots', exist_ok=True)

# Process each region
for city, file in files.items():
    df = pd.read_csv(file)

    # Parse datetime
    df['Date'] = pd.to_datetime(df['Date'])  # Adjust if column name is different
    df.set_index('Date', inplace=True)
    
    # Ensure irradiance column is numeric
    df['Irradiance'] = pd.to_numeric(df['Irradiance'], errors='coerce')
    df = df.dropna(subset=['Irradiance'])

    # Daily average (already daily, so just smooth with rolling if needed)
    daily = df['Irradiance'].resample('D').mean()

    # Weekly average
    weekly = df['Irradiance'].resample('W').mean()

    # Monthly average
    monthly = df['Irradiance'].resample('M').mean()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(monthly, label='Monthly Avg Irradiance')
    plt.title(f'{city} - Monthly Average Irradiance (1950–2020)')
    plt.xlabel('Year')
    plt.ylabel('Irradiance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{city}_monthly_irradiance.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(weekly, label='Weekly Avg Irradiance', color='orange')
    plt.title(f'{city} - Weekly Average Irradiance (1950–2020)')
    plt.xlabel('Year')
    plt.ylabel('Irradiance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{city}_weekly_irradiance.png')
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(daily.rolling(30).mean(), label='Smoothed Daily Avg (30-day MA)', color='green')
    plt.title(f'{city} - Daily Average Irradiance (Smoothed)')
    plt.xlabel('Year')
    plt.ylabel('Irradiance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{city}_daily_irradiance.png')
    plt.close()

print("Plots saved in 'plots/' directory.")

# Save significance explanation
explanation = """
Significance of Annual Irradiance Cycles in Solar Forecasting:

Detecting annual cycles of solar irradiance is crucial in solar energy forecasting for the following reasons:

1. **Seasonal Forecast Accuracy**:
   Cameroon experiences distinct dry and wet seasons. Irradiance levels typically peak during dry months (Nov–Feb) and fall during rainy months (May–Sept). Understanding these cycles allows models to capture seasonality, improving prediction accuracy.

2. **Energy Planning**:
   For solar panel installations and energy resource planning, knowledge of high vs. low irradiance periods is vital to ensure stable power supply throughout the year.

3. **System Sizing and Optimization**:
   Systems can be sized appropriately based on expected irradiance lows, ensuring efficient energy storage and avoiding power shortages.

4. **Policy and Investment Decisions**:
   Long-term irradiance trends help determine viable regions for solar energy projects and inform government or investor decisions.

5. **Climate Change Insights**:
   Comparing cycles over decades (1950–2020) may reveal shifts in seasonal patterns, providing insight into the impact of climate variability on solar resources.

By analyzing monthly and weekly averages, one can clearly visualize these cycles and detect consistent dry-season irradiance peaks and wet-season troughs across all four towns.
"""

with open('irradiance_cycle_significance.txt', 'w') as f:
    f.write(explanation.strip())

print("Significance explanation saved as 'irradiance_cycle_significance.txt'")
