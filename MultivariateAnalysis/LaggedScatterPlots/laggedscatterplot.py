import os
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 1. Configuration ---
cleaned_dir = '../../CleanedDataset'
output_folder = 'laggedscatterplot_plots'
os.makedirs(output_folder, exist_ok=True)

target_variable = 'irradiance'
predictor_variables = ["temperature", "humidity", "potential"]
lags = [1, 7, 365, 1825]  # Lag steps

# --- 2. Load and Process Each Town Dataset ---
csv_files = [f for f in os.listdir(cleaned_dir) if f.endswith('.csv')]

for file in csv_files:
    file_path = os.path.join(cleaned_dir, file)
    try:
        df = pd.read_csv(file_path)
        town_name = os.path.splitext(file)[0]

        # Parse date if available
        if 'date' in df.columns:
            df['date'] = df['date'].astype(str)
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
            df = df.dropna(subset=['date'])
            df.set_index('date', inplace=True)

        # Ensure required columns exist
        required_cols = [target_variable] + predictor_variables
        available_cols = [col for col in required_cols if col in df.columns]
        if target_variable not in df.columns:
            print(f"⚠ Skipping {town_name}: 'irradiance' missing.")
            continue

        # --- 3. Generate Lagged Scatter Subplots ---
        for predictor in predictor_variables:
            if predictor not in df.columns:
                continue

            n_lags = len(lags)
            n_cols = math.ceil(len(lags) / 2)
            fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 9), sharey=True)
            axes = axes.flatten() 
            fig.suptitle(f"{town_name}: Irradiance vs. {predictor} (Lagged)", fontsize=16)

            valid_plot = False
            for i, lag in enumerate(lags):
                lagged_col = f"{predictor}_t-{lag}"
                df[lagged_col] = df[predictor].shift(lag)

                plot_df = df[[target_variable, lagged_col]].dropna()
                if plot_df.empty:
                    axes[i].set_title(f"No Data for t-{lag}")
                    axes[i].axis('off')
                    continue

                sns.scatterplot(
                    x=plot_df[lagged_col],
                    y=plot_df[target_variable],
                    ax=axes[i],
                    alpha=0.6
                )
                axes[i].set_title(f"{predictor}(t-{lag})")
                axes[i].set_xlabel(f"{predictor} (t-{lag})")
                axes[i].set_ylabel("Irradiance" if i == 0 else "")
                axes[i].grid(True, linestyle='--', alpha=0.5)
                valid_plot = True

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if valid_plot:
                lagged_filename = f"{town_name}_lagged_{predictor}_vs_irradiance.png"
                plt.savefig(os.path.join(output_folder, lagged_filename))
                print(f"✅ Saved lagged subplots: {lagged_filename}")
            plt.close()

        # --- 4. PCA (Dimensionality Reduction) ---
        pca_cols = [col for col in predictor_variables if col in df.columns]
        if len(pca_cols) < 2:
            print(f"⚠ Skipping PCA for {town_name}: Not enough predictors.")
            continue

        df_pca = df[pca_cols].dropna()
        if df_pca.empty:
            continue

        X_scaled = StandardScaler().fit_transform(df_pca)
        pca = PCA(n_components=min(len(pca_cols), 3))
        components = pca.fit_transform(X_scaled)

        plt.figure(figsize=(6, 4))
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7)
        plt.ylabel('Explained Variance Ratio')
        plt.xlabel('Principal Components')
        plt.title(f'{town_name}: PCA Explained Variance')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        pca_filename = f"{town_name}_pca_variance.png"
        plt.savefig(os.path.join(output_folder, pca_filename))
        plt.close()
        print(f"✅ Saved PCA plot: {pca_filename}")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
