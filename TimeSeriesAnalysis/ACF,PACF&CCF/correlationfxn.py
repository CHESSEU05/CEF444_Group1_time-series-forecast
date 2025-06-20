import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. Setup
cleaned_dir = "../../CleanedDataset"
files = glob.glob(os.path.join(cleaned_dir, "*.csv"))
vars_ccf = ["temperature", "humidity", "potential", "wind_speed"]
max_acf_lag = 730
max_ccf_lag = 60

sns.set(style="whitegrid")
out_dir = "acf_pacf_ccf_plots"
os.makedirs(out_dir, exist_ok=True)

for file in files:
    town = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    irr = df["irradiance"].dropna()

    # ——— ACF & PACF ———
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(irr, lags=max_acf_lag, ax=axes[0])
    axes[0].set_title(f"{town} Irradiance ACF")
    plot_pacf(irr, lags=max_acf_lag, ax=axes[1], method="ywm")
    axes[1].set_title(f"{town} Irradiance PACF")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{town}_acf_pacf.png"), dpi=300)
    plt.close(fig)

    # ——— Cross‐Correlation Function ———
    lags = range(-max_ccf_lag, max_ccf_lag + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    for var in vars_ccf:
        series = df[var]
        # compute CCF at each lag via correlation of irr[t] vs series[t+lag]
        ccf_vals = [
            irr.corr(series.shift(lag)) 
            for lag in lags
        ]
        ax.plot(lags, ccf_vals, label=var)
    ax.axvline(0, color="k", linestyle="--")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title(f"{town} Irradiance vs Predictors CCF")
    ax.legend()
    ax.grid(axis="both", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{town}_ccf.png"), dpi=300)
    plt.close(fig)

print(f"Saved ACF/PACF and CCF plots to '{out_dir}'")