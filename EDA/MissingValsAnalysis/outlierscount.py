import os
import glob
import pandas as pd
import numpy as np

def detect_outliers_iqr(series, multiplier=1.5):
    """Detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers, lower_bound, upper_bound

def main():
    # 1. Point to your cleaned data
    cleaned_dir = "../../CleanedDataset"
    
    # Check if directory exists
    if not os.path.exists(cleaned_dir):
        print(f"❌ Directory not found: {cleaned_dir}")
        print("Please check the path to your CleanedDataset directory.")
        return
    
    file_paths = glob.glob(os.path.join(cleaned_dir, "*.csv"))
    
    if not file_paths:
        print(f"❌ No CSV files found in: {cleaned_dir}")
        return
    
    print(f"✅ Found {len(file_paths)} CSV files")
    for path in file_paths:
        print(f"  - {os.path.basename(path)}")
    
    # 2. Variables for which to detect outliers
    vars_to_check = ["temperature", "humidity", "irradiance", "potential", "wind_speed"]
    
    # 3. Collect per-town, per-variable outlier stats
    outlier_stats = []
    
    for path in file_paths:
        # Extract town name from filename
        filename = os.path.basename(path)
        town = filename.split('_')[0]  # Get the part before the first underscore
        
        print(f"\n📊 Processing {town}...")
        
        try:
            # Load the data
            df = pd.read_csv(path)
            
            # Check if date column exists and parse it
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            print(f"  Dataset shape: {df.shape}")
            print(f"  Available columns: {list(df.columns)}")
            
            for var in vars_to_check:
                if var in df.columns:
                    series = df[var].dropna()
                    
                    if len(series) > 0:
                        # Compute IQR bounds
                        outliers, lower_bound, upper_bound = detect_outliers_iqr(series)
                        
                        count = len(outliers)
                        total = len(series)
                        percentage = (count / total * 100) if total > 0 else 0
                        
                        outlier_stats.append({
                            "town": town,
                            "variable": var,
                            "outlier_count": count,
                            "total_obs": total,
                            "outlier_percentage": percentage,
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound,
                            "min_value": series.min(),
                            "max_value": series.max()
                        })
                        
                        print(f"    {var}: {count}/{total} outliers ({percentage:.2f}%)")
                    else:
                        print(f"    {var}: No valid data found")
                else:
                    print(f"    {var}: Column not found")
        
        except Exception as e:
            print(f"❌ Error processing {path}: {str(e)}")
            continue
    
    # 4. Build a summary table
    if not outlier_stats:
        print("❌ No outlier statistics collected. Please check your data and column names.")
        return
    
    outlier_df = pd.DataFrame(outlier_stats)
    
    print(f"\n📈 Created outlier DataFrame with shape: {outlier_df.shape}")
    print(f"Columns: {list(outlier_df.columns)}")
    
    # 5. Display summary tables
    print("\n" + "="*80)
    print("OUTLIER SUMMARY (1.5×IQR rule)")
    print("="*80)
    
    try:
        # Outlier counts pivot table
        print("\n📊 OUTLIER COUNTS:")
        outlier_counts = outlier_df.pivot(index="variable", columns="town", values="outlier_count")
        outlier_counts = outlier_counts.fillna(0).astype(int)
        print(outlier_counts)
        
        # Outlier percentages pivot table
        print("\n📊 OUTLIER PERCENTAGES:")
        outlier_percentages = outlier_df.pivot(index="variable", columns="town", values="outlier_percentage")
        outlier_percentages = outlier_percentages.fillna(0).round(2)
        print(outlier_percentages)
        
        # Total observations pivot table
        print("\n📊 TOTAL OBSERVATIONS:")
        total_obs = outlier_df.pivot(index="variable", columns="town", values="total_obs")
        total_obs = total_obs.fillna(0).astype(int)
        print(total_obs)
        
    except Exception as e:
        print(f"❌ Error creating pivot tables: {str(e)}")
        print("\n📋 Raw outlier statistics:")
        print(outlier_df.to_string(index=False))
    
    # 6. Detailed analysis per variable
    print("\n" + "="*80)
    print("DETAILED OUTLIER ANALYSIS")
    print("="*80)
    
    for var in vars_to_check:
        var_data = outlier_df[outlier_df['variable'] == var]
        if not var_data.empty:
            print(f"\n🔍 {var.upper()}:")
            print("-" * 40)
            
            for _, row in var_data.iterrows():
                print(f"📍 {row['town']}:")
                print(f"  • Outliers: {row['outlier_count']:,}/{row['total_obs']:,} ({row['outlier_percentage']:.2f}%)")
                print(f"  • Valid range: {row['lower_bound']:.3f} to {row['upper_bound']:.3f}")
                print(f"  • Actual range: {row['min_value']:.3f} to {row['max_value']:.3f}")
                
                # Identify if there are extreme outliers
                if row['min_value'] < row['lower_bound']:
                    print(f"  ⚠️  Low outliers detected (minimum: {row['min_value']:.3f})")
                if row['max_value'] > row['upper_bound']:
                    print(f"  ⚠️  High outliers detected (maximum: {row['max_value']:.3f})")
                print()
    
    # 7. Summary recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Find variables with high outlier percentages
    high_outlier_vars = outlier_df[outlier_df['outlier_percentage'] > 5.0]
    if not high_outlier_vars.empty:
        print("\n⚠️  VARIABLES WITH HIGH OUTLIER RATES (>5%):")
        for _, row in high_outlier_vars.iterrows():
            print(f"  • {row['town']} - {row['variable']}: {row['outlier_percentage']:.2f}%")
    
    # Find variables with extreme outliers
    extreme_outliers = outlier_df[
        (outlier_df['min_value'] < outlier_df['lower_bound'] - 2 * (outlier_df['upper_bound'] - outlier_df['lower_bound'])) |
        (outlier_df['max_value'] > outlier_df['upper_bound'] + 2 * (outlier_df['upper_bound'] - outlier_df['lower_bound']))
    ]
    
    if not extreme_outliers.empty:
        print("\n🚨 VARIABLES WITH EXTREME OUTLIERS:")
        for _, row in extreme_outliers.iterrows():
            print(f"  • {row['town']} - {row['variable']}")
    
    print("\n✅ Outlier analysis complete!")
    
    # Save results to CSV
    try:
        output_file = "outlier_analysis_results.csv"
        outlier_df.to_csv(output_file, index=False)
        print(f"📁 Results saved to: {output_file}")
    except Exception as e:
        print(f"⚠️  Could not save results: {str(e)}")

if __name__ == "__main__":
    main()