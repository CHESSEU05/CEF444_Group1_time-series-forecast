# CEF444_Group1_time-series-forecast

This repository serves as a workspace dedicated to implementing a robust time-series forecasting system for crucial weather conditions, specifically **solar irradiance**, across several key towns in Cameroon. Accurate irradiance forecasting is vital for optimizing solar energy systems, agricultural planning, and general environmental monitoring.

## 🌟 Project Background

Solar energy is a promising renewable resource, but its variability poses challenges for integration into existing energy grids. Accurate forecasting of solar irradiance (a measure of solar power available at a given location) is essential for efficient energy management, grid stability, and maximizing the utility of solar installations. This project aims to address this need by developing and evaluating time-series models for irradiance prediction in selected Cameroonian towns.

## 🎯 Context and Purpose

The project focuses on building predictive models for solar irradiance in Bafoussam, Bambili, Bamenda, and Yaounde. By leveraging historical weather data, we aim to develop models that can accurately forecast future irradiance levels, thereby supporting better decision-making in solar energy deployment and related sectors.

## 🚀 Objectives

The primary objectives of this project are:

1.  **Data Acquisition & Cleaning:** To gather raw weather data and perform thorough cleaning, including handling missing values and outliers.
2.  **Exploratory Data Analysis (EDA):** To understand the underlying patterns, trends, seasonality, and correlations within the dataset.
3.  **Feature Engineering:** To create new, relevant features from existing data to enhance model performance.
4.  **Model Development:** To implement and train advanced time-series forecasting models (Prophet and SARIMAX).
5.  **Model Evaluation:** To rigorously assess the performance of the developed models against established baselines using standard metrics.
6.  **Visualization & Reporting:** To clearly present model results, insights, and conclusions through comprehensive visualizations and reports.

## 📂 Project Structure

The project is organized into logical directories to ensure clarity, modularity, and ease of navigation:

```.
├── CleanedDataset/                 # Contains the cleaned and preprocessed datasets.
├── DataPreprocessingAndFeatureEngineering/
│   └── OutlierHandledDataset/      # Stores datasets after outlier treatment.
├── Dataset/                        # Stores the raw, uncleaned datasets.
├── EDA/
│   ├── CorrelationAnalysis/        # Scripts and results for correlation analysis.
│   ├── HisBoxPlot_Plots/           # Histograms and box plots for data distribution.
│   ├── MissingValAnalysis/         # Analysis and handling of missing values.
│   └── NumericalStats/             # Numerical summary statistics of the dataset.
├── Irradiance_Analysis/            # Scripts to analyze the correlation of irradiance with temperature,
│                                   # wind speed, humidity, and potential for Bafoussam, Bambili,
│                                   # Bamenda, and Yaounde.
├── ModelSelectionTrainingAndEvaluation/
│   ├── ProphetModel/               # Implementation of the Prophet time-series forecasting model.
│   └── SARIMAXModel/               # Implementation of the SARIMAX time-series forecasting model.
├── MultivariateAnalysis/           # (Further multivariate analysis, e.g., lagged correlations, Granger causality)
├── TimeSeriesAnalysis/             # Scripts to evaluate trends and seasonality in the cleaned dataset.
├── README.md                       # This project README file.
└── rowcleaning.py                  # Script to check and handle empty rows/entries in datasets.
```

## 🛠️ Running the Models

To run the forecasting models and generate predictions and evaluation reports, navigate to the `ModelSelectionTrainingAndEvaluation` directory and execute the respective Python scripts:

### Prophet Model

The Prophet model is designed for ease of use and good performance on time series with strong seasonal effects and holidays.

To run the Prophet model:

```bash
cd ModelSelectionTrainingAndEvaluation/ProphetModel
python prophetmodel.py
SARIMAX Model
The SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) model is a powerful statistical method for time series forecasting that accounts for seasonality and external factors.

To run the SARIMAX model:

Bash

cd ModelSelectionTrainingAndEvaluation/SARIMAXModel
python sarimax_model.py
📊 Results and Visualizations
Upon running the Python scripts in the ModelSelectionTrainingAndEvaluation directory:

Console Output: Key performance metrics (MAE, RMSE, R², etc.) and model summary statistics will be displayed directly in your terminal.

Generated Plots:

For the Prophet model, plots showing the forecast, components (trend, yearly seasonality, weekly seasonality, regressors), and evaluation metrics will be saved as image files (e.g., [TownName]_forecast_only.png, [TownName]_forecast_components.png) in their respective output directories (e.g., prophet_forecast_outputs).

For the SARIMAX model, detailed plots including actual vs. predicted values, residuals analysis, and model diagnostics (ACF, PACF of residuals, Q-Q plot) will be generated and saved as image files (e.g., sarimax_model_evaluation_enhanced.png, sarimax_diagnostics.png).

Summary Reports: Text-based summary reports providing an overview of the forecasting process and results will also be generated (e.g., forecast_summary.txt for Prophet, and a console summary for SARIMAX).

These visualizations and reports are crucial for understanding model performance, identifying patterns, and drawing conclusions.

📝 Conclusion
This project successfully establishes a framework for forecasting solar irradiance in key Cameroonian towns using both Prophet and SARIMAX models. Through comprehensive data preprocessing, feature engineering, and rigorous model evaluation against baselines, we have developed robust forecasting solutions. The generated insights and predictive capabilities contribute significantly to better planning and optimization within the solar energy sector and related climate-dependent activities in the region.

Further work may include exploring more advanced deep learning models, ensemble methods, and real-time data integration for continuous forecasting.
