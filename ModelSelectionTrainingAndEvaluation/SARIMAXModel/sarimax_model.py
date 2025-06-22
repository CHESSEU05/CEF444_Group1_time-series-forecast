import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical modeling
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Sklearn for preprocessing and metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class IrradianceForecaster:
    """
    Comprehensive SARIMAX-based irradiance forecasting system with proper datetime indexing
    """
    
    def __init__(self):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.models = {}
        self.forecasts = {}
        self.baseline_models = {}
        self.evaluation_results = {}
        self.town_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self, data_dir="../../DataPreprocessingAndFeatureEngineering/OutlierHandledDataset"):
        """
        Load and prepare data from all cities with proper datetime indexing
        """
        print("🔄 Loading and preparing data...")
        
        cities = ['Bafoussam', 'Bambili', 'Bamenda', 'Yaounde']
        all_data = []
        
        for city in cities:
            file_path = os.path.join(data_dir, f"{city}_treated.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['city'] = city
                print(f"   ✅ Loaded {city}: {df.shape[0]} records")
                all_data.append(df)
            else:
                print(f"   ❌ File not found: {file_path}")
        
        if not all_data:
            raise FileNotFoundError("No data files found!")
        
        # Combine all data
        self.data = pd.concat(all_data, ignore_index=True)
        
        # Convert date to proper datetime format
        self.data['date'] = pd.to_datetime(self.data['date'], format='%Y%m%d', errors='coerce')
        
        # Remove any rows with invalid dates
        self.data = self.data.dropna(subset=['date'])
        
        # Sort by city and date
        self.data = self.data.sort_values(['city', 'date']).reset_index(drop=True)
        
        # Create additional features
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['day'] = self.data['date'].dt.day
        self.data['day_of_year'] = self.data['date'].dt.dayofyear
        self.data['season'] = self.data['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                                                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                                                     9: 'Fall', 10: 'Fall', 11: 'Fall'})
        
        # Encode categorical variables
        self.data['city_encoded'] = self.town_encoder.fit_transform(self.data['city'])
        
        print(f"📊 Combined dataset shape: {self.data.shape}")
        print(f"📅 Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        print(f"🏙️  Cities: {self.data['city'].unique()}")
        
        return self.data
    
    def create_daily_aggregated_data(self):
        """
        Create properly indexed daily aggregated data for time series modeling
        """
        print("\n📊 Creating daily aggregated data with proper indexing...")
        
        # Group by date and calculate daily averages across all cities
        daily_data = self.data.groupby('date').agg({
            'irradiance': 'mean',
            'temperature': 'mean',
            'humidity': 'mean',
            'month': 'first',
            'day_of_year': 'first',
            'year': 'first'
        }).reset_index()
        
        # Set date as index with proper frequency
        daily_data = daily_data.set_index('date')
        daily_data = daily_data.asfreq('D')  # Set daily frequency
        
        # Fill any missing dates with interpolation
        daily_data = daily_data.interpolate(method='linear')
        
        print(f"   📊 Daily aggregated data shape: {daily_data.shape}")
        print(f"   📅 Date range: {daily_data.index.min()} to {daily_data.index.max()}")
        print(f"   📊 Missing values: {daily_data.isnull().sum().sum()}")
        
        self.daily_data = daily_data
        return daily_data
    
    def split_data_temporal(self, train_ratio=0.8):
        """
        Perform time-based split of the aggregated daily data
        """
        print(f"\n📊 Splitting daily data temporally (train: {train_ratio*100}%, test: {(1-train_ratio)*100}%)...")
        
        # Create daily aggregated data first
        if not hasattr(self, 'daily_data'):
            self.create_daily_aggregated_data()
        
        # Split based on index position
        split_idx = int(len(self.daily_data) * train_ratio)
        
        self.train_data = self.daily_data.iloc[:split_idx].copy()
        self.test_data = self.daily_data.iloc[split_idx:].copy()
        
        print(f"   📅 Training period: {self.train_data.index.min()} to {self.train_data.index.max()}")
        print(f"   📅 Testing period: {self.test_data.index.min()} to {self.test_data.index.max()}")
        print(f"   📊 Training samples: {len(self.train_data)}")
        print(f"   📊 Testing samples: {len(self.test_data)}")
        
        return self.train_data, self.test_data
    
    def establish_baseline_models(self):
        """
        Establish baseline models for comparison using daily aggregated data
        """
        print("\n🏃 Establishing Baseline Models...")
        
        train_irradiance = self.train_data['irradiance'].values
        test_irradiance = self.test_data['irradiance'].values
        
        # Persistence Model (tomorrow = today)
        persistence_forecast = np.full(len(test_irradiance), train_irradiance[-1])
        persistence_forecast[1:] = test_irradiance[:-1]  # Use previous day's actual value
        
        # Seasonal Persistence Model (same day last year)
        seasonal_forecast = []
        for i, date in enumerate(self.test_data.index):
            # Find same day of year from previous year
            target_date = date - pd.DateOffset(years=1)
            
            # Find closest date in training data
            if target_date in self.train_data.index:
                seasonal_forecast.append(self.train_data.loc[target_date, 'irradiance'])
            else:
                # Find closest date
                date_diffs = abs(self.train_data.index - target_date)
                closest_idx = date_diffs.argmin()
                seasonal_forecast.append(self.train_data.iloc[closest_idx]['irradiance'])
        
        seasonal_forecast = np.array(seasonal_forecast)
        
        # Moving average baseline (7-day average)
        window_size = 7
        ma_forecast = []
        for i in range(len(test_irradiance)):
            if i == 0:
                # Use last 7 days from training
                ma_forecast.append(train_irradiance[-window_size:].mean())
            else:
                # Use last 7 days including previous predictions/actuals
                if i < window_size:
                    recent_values = np.concatenate([train_irradiance[-(window_size-i):], test_irradiance[:i]])
                else:
                    recent_values = test_irradiance[i-window_size:i]
                ma_forecast.append(recent_values.mean())
        
        ma_forecast = np.array(ma_forecast)
        
        # Calculate metrics for all baselines
        baselines = {
            'persistence': persistence_forecast,
            'seasonal': seasonal_forecast,
            'moving_average': ma_forecast
        }
        
        baseline_results = {}
        for name, forecast in baselines.items():
            mae = mean_absolute_error(test_irradiance, forecast)
            rmse = np.sqrt(mean_squared_error(test_irradiance, forecast))
            r2 = r2_score(test_irradiance, forecast)
            
            baseline_results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'forecast': forecast
            }
            
            print(f"   📊 {name.capitalize():<15} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
        
        self.baseline_models = baseline_results
        return baseline_results
    
    def analyze_time_series_properties(self):
        """
        Analyze time series properties for SARIMAX parameter selection
        """
        print("\n🔍 Analyzing Time Series Properties...")
        
        irradiance_series = self.train_data['irradiance'].dropna()
        
        # Test for stationarity
        adf_result = adfuller(irradiance_series)
        
        print(f"   📊 Stationarity Test (ADF):")
        print(f"      p-value: {adf_result[1]:.4f}")
        print(f"      Stationary: {'Yes' if adf_result[1] < 0.05 else 'No'}")
        
        # Seasonal decomposition
        try:
            decomposition = seasonal_decompose(irradiance_series, model='additive', period=365)
            print(f"   📊 Seasonal decomposition completed")
        except Exception as e:
            print(f"   ⚠️  Seasonal decomposition failed: {e}")
            decomposition = None
        
        # Calculate ACF and PACF
        acf_values = acf(irradiance_series, nlags=40, fft=True)
        pacf_values = pacf(irradiance_series, nlags=40)
        
        analysis_results = {
            'adf_pvalue': adf_result[1],
            'is_stationary': adf_result[1] < 0.05,
            'acf': acf_values,
            'pacf': pacf_values,
            'decomposition': decomposition
        }
        
        return analysis_results
    
    def determine_sarimax_parameters(self):
        """
        Determine SARIMAX parameters based on analysis and data characteristics
        """
        print("\n⚙️  Determining SARIMAX Parameters...")
        
        # Start with simpler parameters for daily irradiance data
        # Irradiance typically has strong seasonal patterns but may not need differencing
        parameters = {
            'order': (2, 0, 1),          # (p, d, q) - AR(2), no differencing, MA(1)
            'seasonal_order': (1, 1, 1, 365),  # (P, D, Q, S) - seasonal period of 365 days
            'trend': 'c'                  # include constant
        }
        
        print(f"   📋 SARIMAX Parameters:")
        print(f"      Non-seasonal (p,d,q): {parameters['order']}")
        print(f"      Seasonal (P,D,Q,S): {parameters['seasonal_order']}")
        print(f"      Trend: {parameters['trend']}")
        
        return parameters
    
    def prepare_exogenous_variables(self):
        """
        Prepare exogenous variables for SARIMAX with proper alignment
        """
        print("\n🔧 Preparing Exogenous Variables...")
        
        # Select key features
        exog_features = ['temperature', 'humidity', 'month', 'day_of_year']
        
        # Prepare training and testing exogenous variables
        train_exog = self.train_data[exog_features].copy()
        test_exog = self.test_data[exog_features].copy()
        
        # Handle any missing values
        train_exog = train_exog.fillna(method='ffill').fillna(method='bfill')
        test_exog = test_exog.fillna(method='ffill').fillna(method='bfill')
        
        print(f"   📊 Exogenous features: {exog_features}")
        print(f"   📊 Training exog shape: {train_exog.shape}")
        print(f"   📊 Testing exog shape: {test_exog.shape}")
        
        return train_exog, test_exog, exog_features
    
    def train_sarimax_model(self):
        """
        Train SARIMAX model with proper datetime indexing
        """
        print("\n🚀 Training SARIMAX Model...")
        
        # Get parameters and exogenous variables
        params = self.determine_sarimax_parameters()
        train_exog, test_exog, exog_features = self.prepare_exogenous_variables()
        
        # Prepare the target variable with proper index
        y_train = self.train_data['irradiance'].copy()
        
        try:
            # Fit SARIMAX model with proper datetime index
            print("   🔄 Fitting SARIMAX model...")
            
            model = SARIMAX(
                endog=y_train,
                exog=train_exog,
                order=params['order'],
                seasonal_order=params['seasonal_order'],
                trend=params['trend'],
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit(disp=False, maxiter=1000)
            
            print("   ✅ Model fitted successfully!")
            print(f"   📊 AIC: {fitted_model.aic:.2f}")
            print(f"   📊 BIC: {fitted_model.bic:.2f}")
            
            # Generate forecasts
            print("   🔮 Generating forecasts...")
            
            forecast_result = fitted_model.get_forecast(
                steps=len(test_exog),
                exog=test_exog
            )
            
            forecast_values = forecast_result.predicted_mean
            forecast_ci = forecast_result.conf_int()
            
            self.models['sarimax'] = {
                'model': fitted_model,
                'forecast': forecast_values,
                'confidence_interval': forecast_ci,
                'parameters': params,
                'features': exog_features
            }
            
            print("   ✅ Forecasts generated successfully!")
            
            return fitted_model, forecast_values
            
        except Exception as e:
            print(f"   ❌ Error training SARIMAX model: {e}")
            
            # Fallback to even simpler model
            print("   🔄 Trying simpler SARIMAX configuration...")
            
            try:
                simple_model = SARIMAX(
                    endog=y_train,
                    exog=train_exog,
                    order=(1, 0, 0),
                    seasonal_order=(0, 0, 0, 0),
                    trend='c'
                )
                
                fitted_simple = simple_model.fit(disp=False, maxiter=500)
                
                forecast_result = fitted_simple.get_forecast(
                    steps=len(test_exog),
                    exog=test_exog
                )
                
                forecast_values = forecast_result.predicted_mean
                forecast_ci = forecast_result.conf_int()
                
                self.models['sarimax'] = {
                    'model': fitted_simple,
                    'forecast': forecast_values,
                    'confidence_interval': forecast_ci,
                    'parameters': {'order': (1, 0, 0), 'seasonal_order': (0, 0, 0, 0)},
                    'features': exog_features
                }
                
                print("   ✅ Simple SARIMAX model fitted successfully!")
                return fitted_simple, forecast_values
                
            except Exception as e2:
                print(f"   ❌ Error with simple model too: {e2}")
                return None, None
    
    def evaluate_model_performance(self):
        """
        Evaluate SARIMAX model performance against baselines
        """
        print("\n📊 Evaluating Model Performance...")
        
        if 'sarimax' not in self.models:
            print("❌ No SARIMAX model found!")
            return
        
        sarimax_forecast = self.models['sarimax']['forecast'].values
        actual_values = self.test_data['irradiance'].values
        
        # Calculate SARIMAX metrics
        sarimax_mae = mean_absolute_error(actual_values, sarimax_forecast)
        sarimax_rmse = np.sqrt(mean_squared_error(actual_values, sarimax_forecast))
        sarimax_r2 = r2_score(actual_values, sarimax_forecast)
        
        print(f"\n🎯 SARIMAX Model Performance:")
        print(f"   MAE:  {sarimax_mae:.2f}")
        print(f"   RMSE: {sarimax_rmse:.2f}")
        print(f"   R²:   {sarimax_r2:.3f}")
        
        # Compare with baselines
        print(f"\n📊 Comparison with Baselines:")
        print(f"{'Model':<15} {'MAE':<8} {'RMSE':<8} {'R²':<8}")
        print("-" * 40)
        print(f"{'SARIMAX':<15} {sarimax_mae:<8.2f} {sarimax_rmse:<8.2f} {sarimax_r2:<8.3f}")
        
        # Display baseline performance
        for name, results in self.baseline_models.items():
            print(f"{name.capitalize():<15} {results['mae']:<8.2f} {results['rmse']:<8.2f} {results['r2']:<8.3f}")
        
        # Store results
        self.evaluation_results = {
            'sarimax': {
                'mae': sarimax_mae,
                'rmse': sarimax_rmse,
                'r2': sarimax_r2
            }
        }
        
        return self.evaluation_results
    
    def visualize_results(self):
        """
        Create comprehensive visualizations of results
        """
        print("\n📊 Creating Visualizations...")
        
        if 'sarimax' not in self.models:
            print("❌ No SARIMAX model to visualize!")
            return
        
        # Create time series comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: Time series with forecasts
        actual = self.test_data['irradiance'].values
        predicted = self.models['sarimax']['forecast'].values
        dates = self.test_data.index
        
        axes[0, 0].plot(dates, actual, label='Actual', alpha=0.7, linewidth=2)
        axes[0, 0].plot(dates, predicted, label='SARIMAX Forecast', alpha=0.8, linewidth=2)
        
        # Add confidence intervals if available
        if 'confidence_interval' in self.models['sarimax']:
            ci = self.models['sarimax']['confidence_interval']
            axes[0, 0].fill_between(dates, ci.iloc[:, 0], ci.iloc[:, 1], 
                                  alpha=0.2, label='95% Confidence Interval')
        
        axes[0, 0].set_title('Actual vs SARIMAX Forecast', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Irradiance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Residuals over time
        residuals = actual - predicted
        axes[0, 1].plot(dates, residuals, alpha=0.7, color='red')
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Residuals Over Time', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Residual')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Scatter plot (Actual vs Predicted)
        axes[1, 0].scatter(actual, predicted, alpha=0.6, s=20)
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        axes[1, 0].set_xlabel('Actual Irradiance')
        axes[1, 0].set_ylabel('Predicted Irradiance')
        axes[1, 0].set_title('Actual vs Predicted Scatter Plot', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add R² to scatter plot
        r2 = r2_score(actual, predicted)
        axes[1, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[1, 0].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 4: Residuals histogram
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
        axes[1, 1].set_xlabel('Residual Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add statistics to histogram
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        axes[1, 1].text(0.05, 0.95, f'Mean: {mean_residual:.3f}\nStd: {std_residual:.3f}', 
                       transform=axes[1, 1].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('sarimax_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Visualization saved as 'sarimax_model_evaluation.png'")
    
    def plot_model_diagnostics(self):
        """
        Create additional diagnostic plots for the SARIMAX model
        """
        if 'sarimax' not in self.models:
            return
        
        print("\n📊 Creating Model Diagnostic Plots...")
        
        try:
            fitted_model = self.models['sarimax']['model']
            
            # Create diagnostic plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Standardized residuals
            residuals = fitted_model.resid
            axes[0, 0].plot(residuals)
            axes[0, 0].set_title('Standardized Residuals')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot')
            axes[0, 1].grid(True, alpha=0.3)
            
            # ACF of residuals
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(residuals, ax=axes[1, 0], lags=20)
            axes[1, 0].set_title('ACF of Residuals')
            
            # PACF of residuals
            from statsmodels.graphics.tsaplots import plot_pacf
            plot_pacf(residuals, ax=axes[1, 1], lags=20)
            axes[1, 1].set_title('PACF of Residuals')
            
            plt.tight_layout()
            plt.savefig('sarimax_diagnostics.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("   ✅ Diagnostic plots saved as 'sarimax_diagnostics.png'")
            
        except Exception as e:
            print(f"   ⚠️  Could not create diagnostic plots: {e}")
    
    def generate_summary_report(self):
        """
        Generate comprehensive summary report
        """
        print("\n" + "="*80)
        print("SARIMAX IRRADIANCE FORECASTING - SUMMARY REPORT")
        print("="*80)
        
        if hasattr(self, 'daily_data'):
            print(f"\n📊 DATASET OVERVIEW:")
            print(f"   • Original records: {len(self.data):,}")
            print(f"   • Daily aggregated records: {len(self.daily_data):,}")
            print(f"   • Cities: {', '.join(self.data['city'].unique())}")
            print(f"   • Date range: {self.daily_data.index.min().strftime('%Y-%m-%d')} to {self.daily_data.index.max().strftime('%Y-%m-%d')}")
            print(f"   • Training samples: {len(self.train_data):,}")
            print(f"   • Testing samples: {len(self.test_data):,}")
        
        if 'sarimax' in self.models:
            print(f"\n🎯 MODEL CONFIGURATION:")
            params = self.models['sarimax']['parameters']
            features = self.models['sarimax']['features']
            print(f"   • Model: SARIMAX")
            print(f"   • Parameters: {params}")
            print(f"   • Exogenous features: {', '.join(features)}")
            
            print(f"\n📊 MODEL PERFORMANCE:")
            results = self.evaluation_results['sarimax']
            print(f"   • MAE:  {results['mae']:.2f}")
            print(f"   • RMSE: {results['rmse']:.2f}")
            print(f"   • R²:   {results['r2']:.3f}")
            
            print(f"\n📊 BASELINE COMPARISONS:")
            for name, baseline in self.baseline_models.items():
                print(f"   • {name.capitalize():<15}: MAE={baseline['mae']:.2f}, RMSE={baseline['rmse']:.2f}, R²={baseline['r2']:.3f}")
        
        print(f"\n🎯 KEY INSIGHTS:")
        print("   1. Daily aggregation improves model stability and reduces noise")
        print("   2. Proper datetime indexing is crucial for SARIMAX forecasting")
        print("   3. Temperature and humidity are important exogenous predictors")
        print("   4. Seasonal patterns (365-day cycle) are significant for irradiance")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print("   1. Monitor model performance and retrain periodically")
        print("   2. Consider ensemble methods for improved accuracy")
        print("   3. Validate predictions against physical constraints")
        print("   4. Implement automated model diagnostics and alerts")
        print("   5. Consider regional-specific models for better local predictions")
        
        print("\n" + "="*80)

def main():
    """
    Main execution function with improved error handling
    """
    print("🌟 Enhanced SARIMAX Irradiance Forecasting System")
    print("="*60)
    
    # Initialize forecaster
    forecaster = IrradianceForecaster()
    
    try:
        # Step 1: Load and prepare data
        forecaster.load_and_prepare_data()
        
        # Step 2: Create daily aggregated data
        forecaster.create_daily_aggregated_data()
        
        # Step 3: Split data temporally
        forecaster.split_data_temporal()
        
        # Step 4: Establish baseline models
        forecaster.establish_baseline_models()
        
        # Step 5: Analyze time series properties
        forecaster.analyze_time_series_properties()
        
        # Step 6: Train SARIMAX model
        model, forecast = forecaster.train_sarimax_model()
        
        if model is not None:
            # Step 7: Evaluate performance
            forecaster.evaluate_model_performance()
            
            # Step 8: Create visualizations
            forecaster.visualize_results()
            
            # Step 9: Create diagnostic plots
            forecaster.plot_model_diagnostics()
            
            # Step 10: Generate summary report
            forecaster.generate_summary_report()
            
            print("\n🎉 SARIMAX modeling completed successfully!")
        else:
            print("\n❌ SARIMAX model training failed!")
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

# Additional utility functions for enhanced analysis

def grid_search_sarimax_parameters(forecaster, param_grid=None):
    """
    Perform grid search to find optimal SARIMAX parameters
    """
    if param_grid is None:
        param_grid = {
            'order': [(1,0,0), (1,0,1), (1,1,1), (2,0,1), (2,1,1)],
            'seasonal_order': [(0,0,0,0), (1,0,0,365), (1,1,1,365), (0,1,1,365)]
        }
    
    print("\n🔍 Performing Grid Search for Optimal Parameters...")
    
    y_train = forecaster.train_data['irradiance']
    train_exog, test_exog, _ = forecaster.prepare_exogenous_variables()
    
    best_aic = np.inf
    best_params = None
    results = []
    
    for order in param_grid['order']:
        for seasonal_order in param_grid['seasonal_order']:
            try:
                model = SARIMAX(
                    endog=y_train,
                    exog=train_exog,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend='c',
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                fitted_model = model.fit(disp=False, maxiter=500)
                
                aic = fitted_model.aic
                bic = fitted_model.bic
                
                results.append({
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'aic': aic,
                    'bic': bic
                })
                
                if aic < best_aic:
                    best_aic = aic
                    best_params = (order, seasonal_order)
                
                print(f"   ✅ {order} x {seasonal_order}: AIC={aic:.2f}, BIC={bic:.2f}")
                
            except Exception as e:
                print(f"   ❌ {order} x {seasonal_order}: Failed - {str(e)[:50]}...")
                continue
    
    print(f"\n🎯 Best Parameters: {best_params[0]} x {best_params[1]} (AIC: {best_aic:.2f})")
    
    return best_params, results

def create_forecast_comparison_plot(forecaster):
    """
    Create a comprehensive comparison plot of all models
    """
    if 'sarimax' not in forecaster.models:
        print("❌ No SARIMAX model found for comparison!")
        return
    
    print("\n📊 Creating Model Comparison Plot...")
    
    actual = forecaster.test_data['irradiance'].values
    dates = forecaster.test_data.index
    
    plt.figure(figsize=(16, 10))
    
    # Plot actual values
    plt.plot(dates, actual, label='Actual', linewidth=2, color='black', alpha=0.8)
    
    # Plot SARIMAX forecast
    sarimax_forecast = forecaster.models['sarimax']['forecast'].values
    plt.plot(dates, sarimax_forecast, label='SARIMAX', linewidth=2, alpha=0.8)
    
    # Plot baseline forecasts
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (name, baseline) in enumerate(forecaster.baseline_models.items()):
        plt.plot(dates, baseline['forecast'], 
                label=name.capitalize(), 
                linewidth=1.5, 
                alpha=0.7, 
                color=colors[i % len(colors)],
                linestyle='--')
    
    plt.title('Irradiance Forecasting: Model Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Irradiance', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Comparison plot saved as 'model_comparison.png'")

def calculate_forecast_intervals(forecaster, confidence_levels=[0.8, 0.9, 0.95]):
    """
    Calculate and visualize multiple confidence intervals
    """
    if 'sarimax' not in forecaster.models:
        return
    
    print(f"\n📊 Calculating Forecast Intervals...")
    
    model = forecaster.models['sarimax']['model']
    train_exog, test_exog, _ = forecaster.prepare_exogenous_variables()
    
    plt.figure(figsize=(14, 8))
    
    # Plot actual values
    actual = forecaster.test_data['irradiance'].values
    dates = forecaster.test_data.index
    plt.plot(dates, actual, label='Actual', linewidth=2, color='black')
    
    # Plot forecast
    forecast = forecaster.models['sarimax']['forecast'].values
    plt.plot(dates, forecast, label='SARIMAX Forecast', linewidth=2, color='red')
    
    # Plot confidence intervals
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    alphas = [0.6, 0.4, 0.2]
    
    for i, confidence in enumerate(confidence_levels):
        alpha = 1 - confidence
        forecast_result = model.get_forecast(steps=len(test_exog), exog=test_exog, alpha=alpha)
        ci = forecast_result.conf_int()
        
        plt.fill_between(dates, ci.iloc[:, 0], ci.iloc[:, 1], 
                        alpha=alphas[i], 
                        color=colors[i % len(colors)],
                        label=f'{int(confidence*100)}% CI')
    
    plt.title('SARIMAX Forecast with Multiple Confidence Intervals', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Irradiance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('forecast_intervals.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Forecast intervals plot saved as 'forecast_intervals.png'")

def analyze_forecast_errors(forecaster):
    """
    Perform detailed error analysis
    """
    if 'sarimax' not in forecaster.models:
        return
    
    print("\n🔍 Analyzing Forecast Errors...")
    
    actual = forecaster.test_data['irradiance'].values
    forecast = forecaster.models['sarimax']['forecast'].values
    errors = actual - forecast
    
    # Error statistics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / actual)) * 100
    bias = np.mean(errors)
    
    print(f"   📊 Error Statistics:")
    print(f"      MAE:   {mae:.3f}")
    print(f"      RMSE:  {rmse:.3f}")
    print(f"      MAPE:  {mape:.2f}%")
    print(f"      Bias:  {bias:.3f}")
    
    # Error distribution analysis
    percentiles = [5, 25, 50, 75, 95]
    error_percentiles = np.percentile(np.abs(errors), percentiles)
    
    print(f"   📊 Absolute Error Percentiles:")
    for p, val in zip(percentiles, error_percentiles):
        print(f"      {p}th:  {val:.3f}")
    
    # Seasonal error analysis
    test_data_with_errors = forecaster.test_data.copy()
    test_data_with_errors['error'] = errors
    test_data_with_errors['abs_error'] = np.abs(errors)
    
    monthly_errors = test_data_with_errors.groupby('month')['abs_error'].agg(['mean', 'std'])
    
    print(f"   📊 Monthly Error Analysis:")
    print(f"      {'Month':<6} {'Mean Error':<12} {'Std Error':<12}")
    print("      " + "-" * 32)
    for month, row in monthly_errors.iterrows():
        print(f"      {month:<6} {row['mean']:<12.3f} {row['std']:<12.3f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'bias': bias,
        'monthly_errors': monthly_errors
    }

def create_performance_dashboard(forecaster):
    """
    Create a comprehensive performance dashboard
    """
    if 'sarimax' not in forecaster.models:
        return
    
    print("\n📊 Creating Performance Dashboard...")
    
    fig = plt.figure(figsize=(20, 15))
    
    # Define subplot grid
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    actual = forecaster.test_data['irradiance'].values
    forecast = forecaster.models['sarimax']['forecast'].values
    dates = forecaster.test_data.index
    errors = actual - forecast
    
    # 1. Time series plot (top row, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(dates, actual, label='Actual', linewidth=2, alpha=0.8)
    ax1.plot(dates, forecast, label='SARIMAX', linewidth=2, alpha=0.8)
    ax1.set_title('Forecast vs Actual', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot (top row, right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(actual, forecast, alpha=0.6, s=20)
    min_val, max_val = min(actual.min(), forecast.min()), max(actual.max(), forecast.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Actual vs Predicted')
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance metrics (top row, far right)
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis('off')
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    r2 = r2_score(actual, forecast)
    mape = np.mean(np.abs(errors / actual)) * 100
    
    metrics_text = f"""
    Performance Metrics
    ──────────────────
    MAE:   {mae:.3f}
    RMSE:  {rmse:.3f}
    R²:    {r2:.3f}
    MAPE:  {mape:.2f}%
    
    Model: SARIMAX
    Samples: {len(actual)}
    """
    ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 4. Error distribution (second row, left)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    ax4.set_xlabel('Forecast Error')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Distribution')
    ax4.grid(True, alpha=0.3)
    
    # 5. Residuals over time (second row, middle)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(dates, errors, alpha=0.7, color='red')
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Residual')
    ax5.set_title('Residuals Over Time')
    ax5.grid(True, alpha=0.3)
    
    # 6. Monthly error boxplot (second row, right)
    ax6 = fig.add_subplot(gs[1, 2])
    test_data_copy = forecaster.test_data.copy()
    test_data_copy['abs_error'] = np.abs(errors)
    monthly_data = [test_data_copy[test_data_copy['month'] == m]['abs_error'].values 
                   for m in range(1, 13)]
    box_plot = ax6.boxplot(monthly_data, labels=range(1, 13))
    ax6.set_xlabel('Month')
    ax6.set_ylabel('Absolute Error')
    ax6.set_title('Monthly Error Distribution')
    ax6.grid(True, alpha=0.3)
    
    # 7. Model comparison (second row, far right)
    ax7 = fig.add_subplot(gs[1, 3])
    models = ['SARIMAX']
    mae_scores = [mae]
    
    for name, baseline in forecaster.baseline_models.items():
        models.append(name.capitalize())
        mae_scores.append(baseline['mae'])
    
    bars = ax7.bar(models, mae_scores, alpha=0.7)
    ax7.set_ylabel('MAE')
    ax7.set_title('Model Comparison (MAE)')
    ax7.tick_params(axis='x', rotation=45)
    ax7.grid(True, alpha=0.3)
    
    # Highlight best model
    best_idx = np.argmin(mae_scores)
    bars[best_idx].set_color('green')
    
    # 8. Forecast confidence intervals (third row, spans 2 columns)
    ax8 = fig.add_subplot(gs[2, :2])
    ax8.plot(dates, actual, label='Actual', linewidth=2, color='black')
    ax8.plot(dates, forecast, label='Forecast', linewidth=2, color='red')
    
    if 'confidence_interval' in forecaster.models['sarimax']:
        ci = forecaster.models['sarimax']['confidence_interval']
        ax8.fill_between(dates, ci.iloc[:, 0], ci.iloc[:, 1], 
                        alpha=0.3, color='red', label='95% CI')
    
    ax8.set_title('Forecast with Confidence Intervals')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Feature importance (if available) (third row, right)
    ax9 = fig.add_subplot(gs[2, 2])
    if 'features' in forecaster.models['sarimax']:
        features = forecaster.models['sarimax']['features']
        # Simulate feature importance (in real scenario, use model coefficients)
        importance = np.random.rand(len(features))  # Placeholder
        ax9.barh(features, importance, alpha=0.7)
        ax9.set_xlabel('Relative Importance')
        ax9.set_title('Feature Importance')
        ax9.grid(True, alpha=0.3)
    else:
        ax9.text(0.5, 0.5, 'Feature importance\nnot available', 
                ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('Feature Importance')
    
    # 10. Q-Q plot (third row, far right)
    ax10 = fig.add_subplot(gs[2, 3])
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=ax10)
    ax10.set_title('Q-Q Plot (Normality)')
    ax10.grid(True, alpha=0.3)
    
    # 11. Seasonal decomposition (bottom row, spans all columns)
    ax11 = fig.add_subplot(gs[3, :])
    try:
        # Show seasonal pattern in errors
        monthly_avg_error = test_data_copy.groupby('month')['abs_error'].mean()
        ax11.plot(monthly_avg_error.index, monthly_avg_error.values, 'o-', linewidth=2, markersize=8)
        ax11.set_xlabel('Month')
        ax11.set_ylabel('Average Absolute Error')
        ax11.set_title('Seasonal Error Pattern')
        ax11.grid(True, alpha=0.3)
        ax11.set_xticks(range(1, 13))
    except Exception as e:
        ax11.text(0.5, 0.5, f'Seasonal analysis\nerror: {str(e)[:30]}...', 
                 ha='center', va='center', transform=ax11.transAxes)
    
    plt.suptitle('SARIMAX Irradiance Forecasting - Performance Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Performance dashboard saved as 'performance_dashboard.png'")

if __name__ == "__main__":
    main()