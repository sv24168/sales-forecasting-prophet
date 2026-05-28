import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# 1. GENERATE SYNTHETIC SALES DATA

def generate_sales_data():
    """Generates 3 years of daily sales data with trend and seasonality."""
    print("--> Generating synthetic sales data...")
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")
    
    # Base trend (gradual growth)
    trend = np.linspace(100, 250, len(dates))
    
    # Weekly seasonality (higher sales on weekends)
    day_of_week = dates.dayofweek
    weekly_pattern = np.array([10, 12, 15, 20, 35, 50, 40]) # Mon-Sun
    weekly_seasonality = weekly_pattern[day_of_week]
    
    # Yearly seasonality (peak during summer and December holidays)
    month = dates.month
    yearly_seasonality = 30 * np.sin(2 * np.pi * month / 12) + 50 * (month == 12)
    
    # Random Noise
    noise = np.random.normal(0, 15, len(dates))
    
    # Combine components into final sales figures
    sales = trend + weekly_seasonality + yearly_seasonality + noise
    sales = np.clip(sales, 10, None)  # Ensure no negative sales
    
    df = pd.DataFrame({'Date': dates, 'Sales': sales})
    return df

# Initialize data
df = generate_sales_data()
print(df.head())

# 2. EXPLORATORY DATA ANALYSIS (EDA)
print("\n--> Performing EDA...")
# Quick data check
print(df.info())

# Plot historical data
plt.figure(figsize=(14, 5))
plt.plot(df['Date'], df['Sales'], label='Daily Sales', color='royalblue', alpha=0.6)
plt.title('Historical Daily Sales (2023 - 2025)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.legend()
plt.tight_layout()
plt.show()


# 3. DATA PREPARATION FOR PROPHET

# Prophet requires specific column names: 'ds' for dates and 'y' for the target variable
prophet_df = df.rename(columns={'Date': 'ds', 'Sales': 'y'})

# Train/Test Split (Use last 90 days for testing, rest for training)
split_date = '2025-10-02'
train_df = prophet_df[prophet_df['ds'] <= split_date]
test_df = prophet_df[prophet_df['ds'] > split_date]

print(f"Training samples: {len(train_df)}, Testing samples: {len(test_df)}")

# 4. MODEL INITIALIZATION & TRAINING
print("\n--> Training Prophet Model...")
# Initialize Prophet with daily, weekly, and yearly seasonality enabled
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False, # We are using daily data, so intra-day seasonality isn't needed
    interval_width=0.95 # 95% uncertainty confidence intervals
)

# Fit the model
model.fit(train_df)
print("Model training complete!")


# 5. FORECASTING FUTURE SALES
print("\n--> Generating Forecast...")
# Create a dataframe stretching 90 days into the future (matching our test set size)
future = model.make_future_dataframe(periods=90, freq='D')

# Predict sales
forecast = model.predict(future)

# Inspect the forecast columns Prophet generates
# yhat = predicted value, yhat_lower/upper = confidence boundaries
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


# 6. MODEL EVALUATION
print("\n--> Evaluating Model Performance...")
# Merge predictions with actual test values
performance_df = test_df.merge(forecast[['ds', 'yhat']], on='ds')

# Calculate Metrics
mae = mean_absolute_error(performance_df['y'], performance_df['yhat'])
rmse = np.sqrt(mean_squared_error(performance_df['y'], performance_df['yhat']))
mape = np.mean(np.abs((performance_df['y'] - performance_df['yhat']) / performance_df['y'])) * 100

print(f"--- Evaluation Metrics (Test Set) ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


# 7. VISUALIZING RESULTS
# Plot 1: Overall Forecast Components (Trend, Weekly, Yearly patterns)
print("\n--> Plotting Time Series Components...")
model.plot_components(forecast)
plt.show()

# Plot 2: Actual vs Predicted Zoomed In on Test Period
plt.figure(figsize=(14, 6))
plt.plot(train_df['ds'], train_df['y'], label='Train (Actual)', color='black', alpha=0.5)
plt.plot(test_df['ds'], test_df['y'], label='Test (Actual)', color='darkorange', linewidth=2)
plt.plot(performance_df['ds'], performance_df['yhat'], label='Forecast', color='teal', linestyle='--', linewidth=2)
plt.fill_between(forecast['ds'].iloc[-90:], forecast['yhat_lower'].iloc[-90:], forecast['yhat_upper'].iloc[-90:], color='teal', alpha=0.15, label='95% Confidence Interval')

plt.title('Sales Forecast vs Actuals', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()