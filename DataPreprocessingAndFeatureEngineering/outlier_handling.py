import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

class OutlierHandler:
    """
    Comprehensive outlier handling for solar irradiance time series data
    """
    
    def __init__(self, method='hybrid'):
        """
        Initialize outlier handler
        method: 'remove', 'cap', 'transform', 'interpolate', 'hybrid'
        """
        self.method = method
        self.outlier_info = {}
        
    def preserve_date_format(self, df):
        """
        Preserve the original date format from the dataset
        """
        if 'date' in df.columns:
            # Check if dates are in YYYYMMDD format (8 digits)
            sample_date = str(df['date'].iloc[0])
            if len(sample_date) == 8 and sample_date.isdigit():
                # Keep the original YYYYMMDD format
                return df
            else:
                # If already in another format, preserve as is
                return df
        return df
        
    def detect_outliers_multiple_methods(self, series, variable_name):
        """Detect outliers using multiple methods"""
        outlier_indices = set()
        
        # Method 1: IQR (1.5x rule)
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        iqr_lower = Q1 - 1.5 * IQR
        iqr_upper = Q3 + 1.5 * IQR
        iqr_outliers = series[(series < iqr_lower) | (series > iqr_upper)].index
        
        # Method 2: Z-score (threshold = 3)
        z_scores = np.abs(stats.zscore(series))
        z_outliers = series[z_scores > 3].index
        
        # Method 3: Modified Z-score (more robust)
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad != 0:  # Avoid division by zero
            modified_z_scores = 0.6745 * (series - median) / mad
            modified_z_outliers = series[np.abs(modified_z_scores) > 3.5].index
        else:
            modified_z_outliers = []
        
        # Combine outliers (consensus approach)
        outlier_indices.update(iqr_outliers)
        outlier_indices.update(z_outliers)
        outlier_indices.update(modified_z_outliers)
        
        return list(outlier_indices), iqr_lower, iqr_upper
    
    def handle_weather_specific_outliers(self, df, variable):
        """Handle outliers specific to weather variables with improved logic"""
        series = df[variable].copy()
        
        if variable == 'humidity':
            # Humidity should be 0-100%, but data might go above 100%
            # More lenient approach for humidity variations
            upper_limit = series.quantile(0.99)  # More lenient for humidity
            lower_limit = series.quantile(0.01)
            
        elif variable in ['irradiance', 'potential']:
            # Solar data: Low values are real (cloudy days), high values might be sensor errors
            # Be more conservative with low outliers, aggressive with high outliers
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            upper_limit = Q3 + 2.0 * IQR  # More aggressive for high values
            lower_limit = max(0, Q1 - 3.0 * IQR)  # More lenient for low values, but >= 0
            
        elif variable == 'temperature':
            # Temperature: Extreme values possible but rare
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            upper_limit = Q3 + 2.5 * IQR
            lower_limit = Q1 - 2.5 * IQR
            
        elif variable == 'wind_speed':
            # Wind speed: Should be >= 0, high values possible during storms
            lower_limit = 0
            upper_limit = series.quantile(0.995)  # Very lenient upper limit
            
        else:
            # Default IQR method
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            upper_limit = Q3 + 1.5 * IQR
            lower_limit = Q1 - 1.5 * IQR
        
        return lower_limit, upper_limit
    
    def apply_outlier_treatment(self, df, city_name):
        """Apply comprehensive outlier treatment with improved handling"""
        df_treated = df.copy()
        treatment_log = []
        
        variables = ['temperature', 'humidity', 'irradiance', 'potential', 'wind_speed']
        
        for var in variables:
            if var not in df.columns:
                continue
                
            original_series = df[var].copy()
            treated_series = original_series.copy()
            
            # Skip if series is empty or all NaN
            if original_series.isna().all() or len(original_series.dropna()) == 0:
                treatment_log.append({
                    'city': city_name,
                    'variable': var,
                    'n_outliers': 0,
                    'method': 'skipped_empty',
                    'before_min': np.nan,
                    'before_max': np.nan,
                    'after_min': np.nan,
                    'after_max': np.nan,
                    'lower_limit': np.nan,
                    'upper_limit': np.nan
                })
                continue
            
            # Get variable-specific limits
            lower_limit, upper_limit = self.handle_weather_specific_outliers(df, var)
            
            # Find outliers
            outlier_mask = (original_series < lower_limit) | (original_series > upper_limit)
            n_outliers = outlier_mask.sum()
            
            if n_outliers == 0:
                treatment_log.append({
                    'city': city_name,
                    'variable': var,
                    'n_outliers': 0,
                    'method': 'none',
                    'before_min': original_series.min(),
                    'before_max': original_series.max(),
                    'after_min': original_series.min(),
                    'after_max': original_series.max(),
                    'lower_limit': lower_limit,
                    'upper_limit': upper_limit
                })
                continue
            
            # Apply treatment based on method
            if self.method == 'remove':
                # Remove rows with outliers
                df_treated = df_treated[~outlier_mask]
                treated_series = treated_series[~outlier_mask]
                method_used = 'removed'
                
            elif self.method == 'cap':
                treated_series = treated_series.clip(lower=lower_limit, upper=upper_limit)
                method_used = 'capped'
                
            elif self.method == 'interpolate':
                treated_series[outlier_mask] = np.nan
                # Fixed: Use linear interpolation instead of time-based
                treated_series = treated_series.interpolate(method='linear', limit_direction='both')
                method_used = 'interpolated'
                
            elif self.method == 'transform':
                # Use log transformation for highly skewed variables
                if var in ['wind_speed'] and (treated_series > 0).all():
                    treated_series = np.log1p(treated_series)
                    method_used = 'log_transformed'
                else:
                    # Use robust scaling
                    scaler = RobustScaler()
                    treated_series = pd.Series(
                        scaler.fit_transform(treated_series.values.reshape(-1, 1)).flatten(),
                        index=treated_series.index
                    )
                    method_used = 'robust_scaled'
                    
            elif self.method == 'hybrid':
                # Hybrid approach: different treatment for different variables
                if var == 'humidity' and city_name == 'Yaounde':
                    # For Yaoundé humidity (highest outlier rate), use interpolation
                    treated_series[outlier_mask] = np.nan
                    # Fixed: Use linear interpolation instead of time-based
                    treated_series = treated_series.interpolate(method='linear', limit_direction='both')
                    method_used = 'interpolated'
                    
                elif var in ['irradiance', 'potential']:
                    # For solar variables, cap extreme highs but interpolate extreme lows
                    high_outliers = original_series > upper_limit
                    low_outliers = original_series < lower_limit
                    
                    treated_series[high_outliers] = upper_limit  # Cap high values
                    treated_series[low_outliers] = np.nan  # Interpolate low values
                    treated_series = treated_series.interpolate(method='linear', limit_direction='both')
                    method_used = 'hybrid_cap_interpolate'
                    
                elif var == 'wind_speed':
                    # For wind speed, cap extreme values
                    treated_series = treated_series.clip(lower=lower_limit, upper=upper_limit)
                    method_used = 'capped'
                    
                else:
                    # For temperature, use interpolation
                    treated_series[outlier_mask] = np.nan
                    treated_series = treated_series.interpolate(method='linear', limit_direction='both')
                    method_used = 'interpolated'
            
            # Update the dataframe
            df_treated[var] = treated_series
            
            # Log the treatment
            treatment_log.append({
                'city': city_name,
                'variable': var,
                'n_outliers': n_outliers,
                'method': method_used,
                'before_min': original_series.min(),
                'before_max': original_series.max(),
                'after_min': treated_series.min(),
                'after_max': treated_series.max(),
                'lower_limit': lower_limit,
                'upper_limit': upper_limit
            })
        
        return df_treated, treatment_log
    
    def visualize_treatment_effects(self, original_df, treated_df, variable, city_name):
        """Visualize the effects of outlier treatment with improved plotting"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Before treatment
        axes[0, 0].hist(original_df[variable].dropna(), bins=50, alpha=0.7, color='red')
        axes[0, 0].set_title(f'{city_name} - {variable} (Before Treatment)')
        axes[0, 0].set_xlabel(variable)
        axes[0, 0].set_ylabel('Frequency')
        
        # After treatment
        axes[0, 1].hist(treated_df[variable].dropna(), bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title(f'{city_name} - {variable} (After Treatment)')
        axes[0, 1].set_xlabel(variable)
        axes[0, 1].set_ylabel('Frequency')
        
        # Box plots comparison
        data_to_plot = [original_df[variable].dropna(), treated_df[variable].dropna()]
        axes[1, 0].boxplot(data_to_plot, labels=['Before', 'After'])
        axes[1, 0].set_title(f'{city_name} - {variable} Box Plot Comparison')
        axes[1, 0].set_ylabel(variable)
        
        # Time series comparison - Fixed date handling to preserve original format
        if 'date' in original_df.columns:
            # Create a sample for visualization (avoid plotting too many points)
            sample_size = min(1000, len(original_df))
            sample_indices = np.random.choice(len(original_df), sample_size, replace=False)
            sample_indices = np.sort(sample_indices)
            
            sample_data = original_df.iloc[sample_indices].copy()
            sample_treated = treated_df.iloc[sample_indices].copy()
            
            # Use indices for plotting to avoid date parsing issues
            axes[1, 1].plot(range(len(sample_data)), sample_data[variable], 'r-', alpha=0.5, label='Before', linewidth=0.5)
            axes[1, 1].plot(range(len(sample_treated)), sample_treated[variable], 'g-', alpha=0.7, label='After', linewidth=0.5)
            axes[1, 1].set_title(f'{city_name} - {variable} Time Series Comparison (Sample)')
            axes[1, 1].set_xlabel('Time Index')
            axes[1, 1].set_ylabel(variable)
            axes[1, 1].legend()
        else:
            # If no date column, show a scatter plot of indices
            sample_size = min(1000, len(original_df))
            indices = np.random.choice(len(original_df), sample_size, replace=False)
            axes[1, 1].scatter(indices, original_df[variable].iloc[indices], alpha=0.5, color='red', s=1, label='Before')
            axes[1, 1].scatter(indices, treated_df[variable].iloc[indices], alpha=0.7, color='green', s=1, label='After')
            axes[1, 1].set_title(f'{city_name} - {variable} Sample Comparison')
            axes[1, 1].set_xlabel('Index')
            axes[1, 1].set_ylabel(variable)
            axes[1, 1].legend()
        
        plt.tight_layout()
        return fig

    def generate_outlier_report(self, treatment_logs):
        """Generate a comprehensive outlier treatment report"""
        df_log = pd.DataFrame(treatment_logs)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE OUTLIER TREATMENT REPORT")
        print("="*80)
        
        # Overall statistics
        total_outliers = df_log['n_outliers'].sum()
        total_variables = len(df_log)
        variables_with_outliers = len(df_log[df_log['n_outliers'] > 0])
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   • Total variables processed: {total_variables}")
        print(f"   • Variables with outliers: {variables_with_outliers}")
        print(f"   • Total outliers detected: {total_outliers:,}")
        print(f"   • Average outliers per variable: {total_outliers/max(total_variables,1):.1f}")
        
        # City-wise breakdown
        print(f"\n🏙️  CITY-WISE BREAKDOWN:")
        for city in df_log['city'].unique():
            city_data = df_log[df_log['city'] == city]
            city_outliers = city_data['n_outliers'].sum()
            print(f"\n   📍 {city}: {city_outliers:,} total outliers")
            
            for _, row in city_data.iterrows():
                if row['n_outliers'] > 0:
                    percentage = (row['n_outliers'] / len(df_log)) * 100 if len(df_log) > 0 else 0
                    print(f"      • {row['variable']}: {row['n_outliers']:,} outliers ({percentage:.1f}%) → {row['method']}")
        
        # Method usage statistics
        print(f"\n🔧 TREATMENT METHODS USED:")
        method_counts = df_log[df_log['n_outliers'] > 0]['method'].value_counts()
        for method, count in method_counts.items():
            print(f"   • {method}: {count} variables")
        
        return df_log

def process_all_cities():
    """Process all cities with comprehensive outlier handling"""
    
    # Configuration
    input_dir = "../CleanedDataset"
    output_dir = "OutlierHandledDataset"
    plots_dir = "outlier_treatment_plots"
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize outlier handler with hybrid method
    handler = OutlierHandler(method='hybrid')
    
    all_treatment_logs = []
    successfully_processed = []
    
    # Process each city
    cities = ['Bafoussam', 'Bambili', 'Bamenda', 'Yaounde']
    
    for city in cities:
        print(f"\n🏙️  Processing {city}...")
        
        # Load data
        file_path = os.path.join(input_dir, f"{city}_IrrPT.csv")
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            
            # Preserve original date format - don't convert to datetime
            df = handler.preserve_date_format(df)
            
            print(f"   📊 Original dataset shape: {df.shape}")
            if 'date' in df.columns:
                print(f"   📅 Date format preserved: {df['date'].iloc[0]} (first entry)")
            
            # Apply outlier treatment
            df_treated, treatment_log = handler.apply_outlier_treatment(df, city)
            all_treatment_logs.extend(treatment_log)
            
            print(f"   ✅ Treated dataset shape: {df_treated.shape}")
            
            # Save treated dataset
            output_path = os.path.join(output_dir, f"{city}_treated.csv")
            df_treated.to_csv(output_path, index=False)
            print(f"   💾 Saved: {output_path}")
            
            # Create visualizations for variables with significant outliers
            high_outlier_vars = [log['variable'] for log in treatment_log 
                               if log['city'] == city and log['n_outliers'] > 50]
            
            for var in high_outlier_vars:
                if var in df.columns and var in df_treated.columns:
                    try:
                        fig = handler.visualize_treatment_effects(df, df_treated, var, city)
                        plot_path = os.path.join(plots_dir, f"{city}_{var}_treatment.png")
                        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        print(f"   📊 Visualization saved: {plot_path}")
                    except Exception as e:
                        print(f"   ⚠️  Could not create visualization for {var}: {e}")
            
            successfully_processed.append(city)
        
        except Exception as e:
            print(f"❌ Error processing {city}: {e}")
            continue
    
    # Generate comprehensive report
    if all_treatment_logs:
        # Save detailed log
        treatment_df = pd.DataFrame(all_treatment_logs)
        summary_path = "outlier_treatment_detailed_log.csv"
        treatment_df.to_csv(summary_path, index=False)
        
        # Generate and display report
        handler.generate_outlier_report(all_treatment_logs)
        
        print(f"\n✅ PROCESSING COMPLETE!")
        print(f"   📁 Treated datasets: {output_dir}")
        print(f"   📊 Visualizations: {plots_dir}")
        print(f"   📋 Detailed log: {summary_path}")
        print(f"   🏙️  Successfully processed: {', '.join(successfully_processed)}")
        
        # Final recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        print("   1. Use the treated datasets for model training")
        print("   2. Review visualizations to validate treatment effectiveness")
        print("   3. Consider the treatment methods when interpreting model results")
        print("   4. Monitor model performance on both original and treated data")
        print("   5. For variables with many outliers, consider additional feature engineering")
        
        return treatment_df
    else:
        print("\n❌ No data processed successfully! Please check your data files.")
        return None

if __name__ == "__main__":
    # Run the comprehensive outlier handling
    print("🚀 Starting Comprehensive Outlier Handling Process...")
    print("="*80)
    
    summary = process_all_cities()
    
    if summary is not None:
        print(f"\n🎉 All processing completed successfully!")
        print("Your data is now ready for time series forecasting.")
    else:
        print(f"\n💥 Processing failed. Please check your data files and directory structure.")