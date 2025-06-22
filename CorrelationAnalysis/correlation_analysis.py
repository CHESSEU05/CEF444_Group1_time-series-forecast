import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_cleaned_data():
    """Load all cleaned CSV files from CleanedDataset directory"""
    data_dir = "../CleanedDataset"
    file_paths = [
        "Bafoussam_IrrPT.csv",
        "Bambili_IrrPT.csv", 
        "Bamenda_IrrPT.csv",
        "Yaounde_IrrPT.csv"
    ]
    
    datasets = {}
    for file in file_paths:
        path = os.path.join(data_dir, file)
        if os.path.exists(path):
            city_name = file.split('_')[0]
            df = pd.read_csv(path)
            
            # Convert date column to datetime if it exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
                df = df.sort_values('date')
            
            datasets[city_name] = df
            print(f"✅ Loaded {city_name}: {len(df)} records")
        else:
            print(f"⚠️ File not found: {path}")
    
    return datasets

def get_numerical_columns(df):
    """Get numerical columns for correlation analysis"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove date-related columns if they exist
    numerical_cols = [col for col in numerical_cols if not any(x in col.lower() for x in ['date', 'year', 'month', 'day'])]
    return numerical_cols

def calculate_correlation_matrix(df, method='pearson'):
    """Calculate correlation matrix using specified method"""
    numerical_cols = get_numerical_columns(df)
    if len(numerical_cols) < 2:
        return None, []
    
    corr_data = df[numerical_cols].dropna()
    if method.lower() == 'pearson':
        corr_matrix = corr_data.corr(method='pearson')
    elif method.lower() == 'spearman':
        corr_matrix = corr_data.corr(method='spearman')
    else:
        corr_matrix = corr_data.corr(method='pearson')
    
    return corr_matrix, numerical_cols

def plot_correlation_heatmap(corr_matrix, city_name, method='Pearson'):
    """Create correlation heatmap"""
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .8})
    
    plt.title(f'{method} Correlation Matrix - {city_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return plt.gcf()

def analyze_irradiance_correlations(df, city_name):
    """Analyze specific correlations with irradiance"""
    if 'irradiance' not in df.columns:
        print(f"⚠️ No irradiance column found in {city_name}")
        return None
    
    numerical_cols = get_numerical_columns(df)
    other_vars = [col for col in numerical_cols if col != 'irradiance']
    
    correlations = {}
    for var in other_vars:
        # Remove NaN values for correlation calculation
        clean_data = df[['irradiance', var]].dropna()
        if len(clean_data) > 5:  # Need at least 5 points for meaningful correlation
            pearson_corr, pearson_p = pearsonr(clean_data['irradiance'], clean_data[var])
            spearman_corr, spearman_p = spearmanr(clean_data['irradiance'], clean_data[var])
            
            correlations[var] = {
                'pearson_r': pearson_corr,
                'pearson_p': pearson_p,
                'spearman_r': spearman_corr,
                'spearman_p': spearman_p,
                'n_samples': len(clean_data)
            }
    
    return correlations

def plot_irradiance_vs_variable(df, city_name, var_name, ax=None):
    """Plot irradiance vs another variable"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    clean_data = df[['irradiance', var_name]].dropna()
    
    if len(clean_data) > 0:
        ax.scatter(clean_data[var_name], clean_data['irradiance'], alpha=0.6, s=30)
        
        # Add trend line
        if len(clean_data) > 2:
            z = np.polyfit(clean_data[var_name], clean_data['irradiance'], 1)
            p = np.poly1d(z)
            ax.plot(clean_data[var_name], p(clean_data[var_name]), "r--", alpha=0.8)
        
        # Calculate and display correlation
        corr, p_val = pearsonr(clean_data['irradiance'], clean_data[var_name])
        ax.set_title(f'{city_name}: Irradiance vs {var_name.title()}\nPearson r = {corr:.3f} (p = {p_val:.3f})')
        ax.set_xlabel(var_name.title())
        ax.set_ylabel('Irradiance')
        ax.grid(True, alpha=0.3)
    
    return ax

def create_comprehensive_plots(datasets):
    """Create comprehensive correlation plots for all cities"""
    
    # 1. Correlation Heatmaps
    print("Creating correlation heatmaps...")
    fig_heatmaps = plt.figure(figsize=(20, 15))
    
    for i, (city, df) in enumerate(datasets.items(), 1):
        # Pearson correlation
        plt.subplot(2, 4, i)
        corr_matrix, _ = calculate_correlation_matrix(df, 'pearson')
        if corr_matrix is not None:
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                       square=True, fmt='.2f', cbar=i==1)
            plt.title(f'{city} - Pearson')
        
        # Spearman correlation
        plt.subplot(2, 4, i+4)
        corr_matrix, _ = calculate_correlation_matrix(df, 'spearman')
        if corr_matrix is not None:
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                       square=True, fmt='.2f', cbar=i==1)
            plt.title(f'{city} - Spearman')
    
    plt.suptitle('Correlation Matrices - All Cities', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmaps_all_cities.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Irradiance vs Humidity plots
    print("Creating irradiance vs humidity plots...")
    fig_humidity = plt.figure(figsize=(16, 12))
    
    for i, (city, df) in enumerate(datasets.items(), 1):
        if 'humidity' in df.columns:
            ax = plt.subplot(2, 2, i)
            plot_irradiance_vs_variable(df, city, 'humidity', ax)
    
    plt.suptitle('Irradiance vs Humidity - All Cities', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('irradiance_vs_humidity_all_cities.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Irradiance vs Temperature plots
    print("Creating irradiance vs temperature plots...")
    fig_temp = plt.figure(figsize=(16, 12))
    
    for i, (city, df) in enumerate(datasets.items(), 1):
        if 'temperature' in df.columns:
            ax = plt.subplot(2, 2, i)
            plot_irradiance_vs_variable(df, city, 'temperature', ax)
    
    plt.suptitle('Irradiance vs Temperature - All Cities', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('irradiance_vs_temperature_all_cities.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Irradiance vs Wind Speed plots (if available)
    print("Creating irradiance vs wind speed plots...")
    wind_cols = []
    for city, df in datasets.items():
        wind_cols.extend([col for col in df.columns if 'wind' in col.lower()])
    
    if wind_cols:
        fig_wind = plt.figure(figsize=(16, 12))
        
        for i, (city, df) in enumerate(datasets.items(), 1):
            wind_col = None
            for col in df.columns:
                if 'wind' in col.lower():
                    wind_col = col
                    break
            
            if wind_col:
                ax = plt.subplot(2, 2, i)
                plot_irradiance_vs_variable(df, city, wind_col, ax)
        
        plt.suptitle('Irradiance vs Wind Speed - All Cities', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('irradiance_vs_windspeed_all_cities.png', dpi=300, bbox_inches='tight')
        plt.show()

def calculate_cross_correlation(df, target_var='irradiance', max_lag=30):
    """Calculate cross-correlation between irradiance and other variables"""
    if target_var not in df.columns:
        return {}
    
    numerical_cols = get_numerical_columns(df)
    other_vars = [col for col in numerical_cols if col != target_var]
    
    cross_correlations = {}
    
    for var in other_vars:
        clean_data = df[[target_var, var]].dropna()
        if len(clean_data) > max_lag * 2:  # Need enough data for lagged analysis
            
            # Calculate cross-correlation at different lags
            lags = range(-max_lag, max_lag + 1)
            correlations = []
            
            for lag in lags:
                if lag == 0:
                    corr, _ = pearsonr(clean_data[target_var], clean_data[var])
                elif lag > 0:
                    # Positive lag: var leads target_var
                    if len(clean_data) > lag:
                        corr, _ = pearsonr(clean_data[target_var][lag:], clean_data[var][:-lag])
                    else:
                        corr = np.nan
                else:  # lag < 0
                    # Negative lag: target_var leads var
                    abs_lag = abs(lag)
                    if len(clean_data) > abs_lag:
                        corr, _ = pearsonr(clean_data[target_var][:-abs_lag], clean_data[var][abs_lag:])
                    else:
                        corr = np.nan
                
                correlations.append(corr)
            
            cross_correlations[var] = {
                'lags': list(lags),
                'correlations': correlations,
                'max_corr': np.nanmax(np.abs(correlations)),
                'best_lag': lags[np.nanargmax(np.abs(correlations))]
            }
    
    return cross_correlations

def plot_cross_correlations(datasets):
    """Plot cross-correlation analysis"""
    print("Calculating and plotting cross-correlations...")
    
    fig = plt.figure(figsize=(20, 15))
    
    plot_idx = 1
    for city, df in datasets.items():
        cross_corrs = calculate_cross_correlation(df)
        
        if cross_corrs:
            for var, corr_data in cross_corrs.items():
                if plot_idx <= 12:  # Limit number of subplots
                    ax = plt.subplot(3, 4, plot_idx)
                    
                    lags = corr_data['lags']
                    correlations = corr_data['correlations']
                    
                    ax.plot(lags, correlations, 'b-', linewidth=2)
                    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
                    
                    # Highlight maximum correlation
                    max_idx = np.nanargmax(np.abs(correlations))
                    ax.plot(lags[max_idx], correlations[max_idx], 'ro', markersize=8)
                    
                    ax.set_title(f'{city}: Irradiance vs {var.title()}\nMax |r| = {corr_data["max_corr"]:.3f} at lag {corr_data["best_lag"]}')
                    ax.set_xlabel('Lag (days)')
                    ax.set_ylabel('Cross-correlation')
                    ax.grid(True, alpha=0.3)
                    
                    plot_idx += 1
    
    plt.suptitle('Cross-Correlation Analysis - Irradiance with Other Variables', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cross_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_correlation_summary(datasets):
    """Generate summary report of correlations"""
    print("\n" + "="*80)
    print("COMPREHENSIVE CORRELATION ANALYSIS SUMMARY")
    print("="*80)
    
    for city, df in datasets.items():
        print(f"\n📍 {city.upper()}")
        print("-" * 40)
        
        # Basic dataset info
        print(f"Dataset size: {len(df)} records")
        numerical_cols = get_numerical_columns(df)
        print(f"Numerical variables: {', '.join(numerical_cols)}")
        
        # Irradiance correlations
        if 'irradiance' in df.columns:
            corr_results = analyze_irradiance_correlations(df, city)
            if corr_results:
                print("\n🌞 IRRADIANCE CORRELATIONS:")
                
                for var, stats in corr_results.items():
                    pearson_r = stats['pearson_r']
                    pearson_p = stats['pearson_p']
                    
                    # Interpret correlation strength
                    if abs(pearson_r) >= 0.7:
                        strength = "Strong"
                    elif abs(pearson_r) >= 0.5:
                        strength = "Moderate"
                    elif abs(pearson_r) >= 0.3:
                        strength = "Weak"
                    else:
                        strength = "Very Weak"
                    
                    # Interpret significance
                    significance = "***" if pearson_p < 0.001 else "**" if pearson_p < 0.01 else "*" if pearson_p < 0.05 else "ns"
                    
                    print(f"  • {var.title()}: r = {pearson_r:+.3f} ({strength}) {significance}")
                    print(f"    Sample size: {stats['n_samples']}")
        
        # Overall correlation matrix summary
        corr_matrix, cols = calculate_correlation_matrix(df, 'pearson')
        if corr_matrix is not None:
            # Find strongest correlations (excluding self-correlations)
            corr_values = []
            for i in range(len(cols)):
                for j in range(i+1, len(cols)):
                    corr_values.append((cols[i], cols[j], corr_matrix.iloc[i, j]))
            
            if corr_values:
                corr_values.sort(key=lambda x: abs(x[2]), reverse=True)
                print(f"\n🔗 STRONGEST OVERALL CORRELATIONS:")
                for var1, var2, corr in corr_values[:3]:  # Top 3
                    print(f"  • {var1.title()} ↔ {var2.title()}: r = {corr:+.3f}")

def main():
    """Main analysis function"""
    print("🌤️  COMPREHENSIVE IRRADIANCE CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Load datasets
    datasets = load_cleaned_data()
    
    if not datasets:
        print("❌ No datasets found. Please run the data cleaning script first.")
        return
    
    # Create output directory for plots
    os.makedirs('correlation_analysis_plots', exist_ok=True)
    os.chdir('correlation_analysis_plots')
    
    try:
        # Perform comprehensive analysis
        create_comprehensive_plots(datasets)
        plot_cross_correlations(datasets)
        
        # Generate summary report
        generate_correlation_summary(datasets)
        
        print(f"\n✅ Analysis complete! Plots saved in 'correlation_analysis_plots' directory.")
        print("\n📊 Key Insights to Look For:")
        print("• Negative correlation between irradiance and humidity (cloud cover effect)")
        print("• Positive correlation between irradiance and temperature (generally)")
        print("• Wind speed effects on irradiance (cloud dispersion)")
        print("• Lagged relationships in cross-correlation analysis")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        os.chdir('..')

if __name__ == "__main__":
    main()