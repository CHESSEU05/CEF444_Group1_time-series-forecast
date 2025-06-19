import os
import pandas as pd

# 1. List your exact CSV paths
file_paths = [
    "Dataset/Bafoussam_IrrPT.csv",
    "Dataset/Bambili_IrrPT.csv",
    "Dataset/Bamenda_IrrPT.csv",
    "Dataset/Yaounde_IrrPT.csv"
]

# 2. Map all known header variants to your canonical names
COLUMN_MAPPING = {
    # date variants
    "date":           "date",
    "Date":           "date",
    # temperature variants
    "Temperature":    "temperature",
    # humidity variants
    "humidity":       "humidity",
    "humudity":       "humidity",
    "Humidity":       "humidity",
    # irradiance variants
    "irradiance":     "irradiance",
    "Irradiance":     "irradiance",
    # potential variants
    "potential":      "potential",
    # wind speed variants
    "wind speed":      "wind speed",
    "Wind_Speed":      "wind speed",
}

def standardize_headers(cols):
    """
    Normalize and rename a list of column names.
    Returns a new list of canonical column names.
    """
    # 1) Normalize: lowercase, trim, replace spaces/dashes with underscore
    clean = (
        pd.Series(cols)
          .str.lower()
          .str.strip()
          .str.replace(r"[ \-\/]", "_", regex=True)
          .str.replace(r"[^0-9a-z_]", "", regex=True)
    )

    # 2) Apply mapping
    mapped = clean.map(lambda c: COLUMN_MAPPING.get(c, c))
    return mapped.tolist()

# 3. Prepare output directory
output_dir = "CleanedDataset"
os.makedirs(output_dir, exist_ok=True)

# 4. Process each file
for path in file_paths:
    if not os.path.isfile(path):
        print(f"⚠️  File not found, skipping: {path}")
        continue

    # read full CSV
    df = pd.read_csv(path)

    # standardize header row
    df.columns = standardize_headers(df.columns)

    # write cleaned CSV including all data rows
    out_path = os.path.join(output_dir, os.path.basename(path))
    df.to_csv(out_path, index=False)
    print(f"✅  Saved cleaned file: {out_path}")

print("\nAll files have been processed; cleaned files are in 'CleanedDataset/'.")
