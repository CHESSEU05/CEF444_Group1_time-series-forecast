import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Define Paths ---
cleaned_dir = '../../CleanedDataset'
output_folder = 'scatterplot_plots'
os.makedirs(output_folder, exist_ok=True)

# --- 2. Define Variables ---
target_variable = 'irradiance'
predictor_variables = ["temperature", "humidity", "potential", "wind_speed"]

# --- 3. Load and Concatenate All Town Datasets ---
all_dataframes = []
csv_files = [f for f in os.listdir(cleaned_dir) if f.endswith('.csv')]

for file in csv_files:
    file_path = os.path.join(cleaned_dir, file)
    try:
        df = pd.read_csv(file_path)
        
        # Parse date
        df['date'] = df['date'].astype(str)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['date'])  # Drop rows with invalid dates
        df = df.set_index('date')
        
        # Assign Town name from filename if not present
        town_name = os.path.splitext(file)[0]
        if 'Town' not in df.columns:
            df['Town'] = town_name
        
        all_dataframes.append(df)
        print(f"✅ Loaded: {file}")
    except Exception as e:
        print(f"❌ Failed to load {file}: {e}")

# Combine all datasets
if not all_dataframes:
    print("❌ No valid datasets found. Exiting.")
    exit()

combined_df = pd.concat(all_dataframes, ignore_index=False)

# --- 4. Generate Scatter Plots: One per Predictor per Town ---
print("\n--- Generating Individual Scatter Plots per Town ---")
towns = combined_df['Town'].unique()

for town in towns:
    town_data = combined_df[combined_df['Town'] == town]
    
    for predictor in predictor_variables:
        if predictor not in town_data.columns:
            print(f"⚠ Column '{predictor}' not found for town '{town}'. Skipping.")
            continue
        try:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=town_data[predictor], y=town_data[target_variable], alpha=0.6)
            plt.title(f'{town} - Scatter Plot: Irradiance vs. {predictor}', fontsize=16)
            plt.xlabel(predictor, fontsize=12)
            plt.ylabel('Irradiance', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            filename = f"{town}_irradiance_vs_{predictor.lower().replace(' ', '_')}_scatter.png"
            save_path = os.path.join(output_folder, filename)
            plt.savefig(save_path)
            plt.close()
            print(f"✅ Saved: {filename}")
        except Exception as e:
            print(f"❌ Error plotting {predictor} for {town}: {e}")

# --- 5. Generate Pair Plots Per Town ---
print("\n--- Generating Pair Plots per Town ---")
for town in towns:
    town_data = combined_df[combined_df['Town'] == town]

    # Check that all required columns are present
    columns_for_plot = [target_variable] + predictor_variables
    missing = [col for col in columns_for_plot if col not in town_data.columns]
    if missing:
        print(f"⚠ Skipping pair plot for {town}. Missing columns: {missing}")
        continue

    try:
        pair_plot = sns.pairplot(town_data[columns_for_plot], diag_kind="kde", plot_kws={"alpha": 0.5})
        pair_plot.fig.suptitle(f'Pair Plot: {town}', y=1.02, fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])

        pairplot_filename = f"{town}_pair_plot.png"
        pairplot_save_path = os.path.join(output_folder, pairplot_filename)
        pair_plot.fig.savefig(pairplot_save_path)
        plt.close(pair_plot.fig)
        print(f"✅ Saved: {pairplot_filename}")
    except Exception as e:
        print(f"❌ Error generating pair plot for {town}: {e}")
