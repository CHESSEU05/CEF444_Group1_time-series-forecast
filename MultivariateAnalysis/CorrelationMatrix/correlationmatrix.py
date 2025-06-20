import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Configuration ---
cleaned_dir = '../../CleanedDataset'
output_folder = 'heatmap_plots'
os.makedirs(output_folder, exist_ok=True)

# Define variables
target_variable = 'irradiance'
predictor_variables = ["temperature", "humidity", "potential", "wind_speed"]
relevant_columns = [target_variable] + predictor_variables

# --- 2. Load and Process Each CSV ---
csv_files = [f for f in os.listdir(cleaned_dir) if f.endswith('.csv')]

for file in csv_files:
    file_path = os.path.join(cleaned_dir, file)
    try:
        df = pd.read_csv(file_path)

        # Extract town name from file name (e.g., "Bambili_IrrPT.csv" → "Bambili_IrrPT")
        town_name = os.path.splitext(file)[0]

        # Parse date column if it exists
        if 'date' in df.columns:
            df['date'] = df['date'].astype(str)
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
            df = df.dropna(subset=['date'])
            df.set_index('date', inplace=True)

        # Select only numeric columns from relevant list
        present_columns = [col for col in relevant_columns if col in df.columns]
        if len(present_columns) < 2:
            print(f"⚠ Skipping {town_name}: Not enough relevant columns for correlation.")
            continue

        df_subset = df[present_columns].select_dtypes(include='number')

        if df_subset.shape[1] < 2:
            print(f"⚠ Skipping {town_name}: Less than two numeric columns available.")
            continue

        # --- 3. Compute Correlations ---
        pearson_corr_matrix = df_subset.corr(method='pearson')
        spearman_corr_matrix = df_subset.corr(method='spearman')

        # --- 4. Pearson Plot Heatmaps ---
        plt.figure(figsize=(8, 6))
        sns.heatmap(pearson_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
        plt.title(f'Pearson Correlation Heatmap: {town_name}', fontsize=16)
        plt.tight_layout()

        # --- 5. Save Pearson Plot ---
        heatmap_filename = f"{town_name}_pearson_correlation_heatmap.png"
        heatmap_path = os.path.join(output_folder, heatmap_filename)
        plt.savefig(heatmap_path)
        plt.close()
        print(f"✅ Saved heatmap: {heatmap_filename}")

        
        # --- 6. Plot Heatmaps ---
        plt.figure(figsize=(8, 6))
        sns.heatmap(spearman_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
        plt.title(f'Spearman Correlation Heatmap: {town_name}', fontsize=16)
        plt.tight_layout()

        # --- 7. Save Plot ---
        heatmap_filename = f"{town_name}_spearman_correlation_heatmap.png"
        heatmap_path = os.path.join(output_folder, heatmap_filename)
        plt.savefig(heatmap_path)
        plt.close()
        print(f"✅ Saved heatmap: {heatmap_filename}")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
