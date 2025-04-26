#!/usr/bin/env python
# SPARCS Dataset Predictive Analysis

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap

# Set random seed for reproducibility
np.random.seed(42)

# Display settings
pd.set_option('display.max_columns', None)
sns.set(style='whitegrid')

print("Step 1: Data Loading & Preparation")
# Load the dataset
file_path = 'Hospital_Inpatient_Discharges__SPARCS_De-Identified___Cost_Transparency__Beginning_2009_20250426.csv'
df = pd.read_csv(file_path)

# Display basic information
print(f"Dataset shape: {df.shape}")
print(df.head())

# Keep only relevant columns based on actual column names in the dataset
relevant_cols = [
    'Year', 
    'Facility Id', 
    'Facility Name', 
    'APR DRG Code',
    'APR Severity of Illness Code', 
    'APR Medical Surgical Code', 
    'Discharges', 
    'Mean Cost', 
    'Median Cost', 
    'Mean Charge', 
    'Median Charge'
]

# Check if all columns exist in the dataset
missing_cols = [col for col in relevant_cols if col not in df.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}")
    # Show actual columns for reference
    print("Available columns:")
    print(df.columns.tolist())
else:
    # Keep only the relevant columns
    df = df[relevant_cols]
    
# Display the filtered dataset
print("Filtered dataset:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values)

# Check data types
print("\nData types:")
print(df.dtypes)

# Convert categorical columns to appropriate data types
categorical_cols = ['Facility Id', 'APR DRG Code', 'APR Severity of Illness Code', 'APR Medical Surgical Code']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Convert numeric columns to appropriate types
numeric_cols = ['Year', 'Discharges', 'Mean Cost', 'Median Cost', 'Mean Charge', 'Median Charge']
for col in numeric_cols:
    if col in df.columns:
        # Handle possible non-numeric characters in the data
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('"', '').str.replace('$', ''), errors='coerce')

# Check data types after conversion
print("Data types after conversion:")
print(df.dtypes)

# Handle missing values
# For numeric columns: use median imputation
# For categorical columns: use mode imputation

# First, check the percentage of missing values in each column
missing_percentage = (df.isnull().sum() / len(df)) * 100
print("Percentage of missing values:")
print(missing_percentage)

# Impute missing values
for col in df.columns:
    if df[col].dtype.name == 'category':
        # For categorical variables, use the most frequent value
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    else:
        # For numeric variables, use the median
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

# Check for any remaining missing values
print("\nRemaining missing values:")
print(df.isnull().sum())

# Check the range of years in the dataset
print(f"Years range from {df['Year'].min()} to {df['Year'].max()}")

# Basic statistics of the dataset
print("\nBasic statistics:")
print(df.describe())

print("\nStep 2: Feature Engineering")
# Create historical aggregates by Facility ID
facility_aggs = df.groupby('Facility Id').agg({
    'Mean Cost': 'mean',
    'Median Cost': 'mean',
    'Mean Charge': 'mean',
    'Median Charge': 'mean',
    'Discharges': 'mean'
}).reset_index()

facility_aggs.columns = [
    'Facility Id', 
    'Facility_Avg_Mean_Cost', 
    'Facility_Avg_Median_Cost', 
    'Facility_Avg_Mean_Charge', 
    'Facility_Avg_Median_Charge',
    'Facility_Avg_Discharges'
]

# Create historical aggregates by DRG Code
drg_aggs = df.groupby('APR DRG Code').agg({
    'Mean Cost': 'mean',
    'Median Cost': 'mean',
    'Mean Charge': 'mean',
    'Median Charge': 'mean',
    'Discharges': 'mean'
}).reset_index()

drg_aggs.columns = [
    'APR DRG Code', 
    'DRG_Avg_Mean_Cost', 
    'DRG_Avg_Median_Cost', 
    'DRG_Avg_Mean_Charge', 
    'DRG_Avg_Median_Charge',
    'DRG_Avg_Discharges'
]

# Create historical aggregates by Severity of Illness Code
severity_aggs = df.groupby('APR Severity of Illness Code').agg({
    'Mean Cost': 'mean',
    'Median Cost': 'mean',
    'Mean Charge': 'mean',
    'Median Charge': 'mean',
    'Discharges': 'mean'
}).reset_index()

severity_aggs.columns = [
    'APR Severity of Illness Code', 
    'Severity_Avg_Mean_Cost', 
    'Severity_Avg_Median_Cost', 
    'Severity_Avg_Mean_Charge', 
    'Severity_Avg_Median_Charge',
    'Severity_Avg_Discharges'
]

# Merge the aggregated features back to the original dataframe
df = pd.merge(df, facility_aggs, on='Facility Id', how='left')
df = pd.merge(df, drg_aggs, on='APR DRG Code', how='left')
df = pd.merge(df, severity_aggs, on='APR Severity of Illness Code', how='left')

# Create Year-over-Year (YoY) changes
# First, create a key combining Facility Id and APR DRG Code
df['Facility_DRG_Key'] = df['Facility Id'].astype(str) + '_' + df['APR DRG Code'].astype(str) + '_' + df['APR Severity of Illness Code'].astype(str)

# Sort by key and year
df = df.sort_values(['Facility_DRG_Key', 'Year'])

# Create YoY changes
df['YoY_Discharges_Change'] = df.groupby('Facility_DRG_Key')['Discharges'].pct_change()
df['YoY_Mean_Cost_Change'] = df.groupby('Facility_DRG_Key')['Mean Cost'].pct_change()
df['YoY_Median_Cost_Change'] = df.groupby('Facility_DRG_Key')['Median Cost'].pct_change()
df['YoY_Mean_Charge_Change'] = df.groupby('Facility_DRG_Key')['Mean Charge'].pct_change()
df['YoY_Median_Charge_Change'] = df.groupby('Facility_DRG_Key')['Median Charge'].pct_change()

# Fill NaN values in YoY changes (first year will have NaN)
yoy_cols = [col for col in df.columns if 'YoY' in col]
df[yoy_cols] = df[yoy_cols].fillna(0)

# Create temporal indicators - year as a categorical feature
df['Year_Cat'] = df['Year'].astype('category')

# One-hot encode categorical variables
print("\nEncoding categorical variables...")
# Convert categorical columns to strings for one-hot encoding
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)

# Apply one-hot encoding to categorical columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_cats = encoder.fit_transform(df[categorical_cols])

# Get feature names after one-hot encoding
feature_names = []
for i, col in enumerate(categorical_cols):
    categories = encoder.categories_[i][1:]  # Skip the first category (dropped)
    feature_names.extend([f"{col}_{cat}" for cat in categories])

# Create a DataFrame with the encoded features
encoded_df = pd.DataFrame(encoded_cats, columns=feature_names, index=df.index)

# Save original columns needed for task 2 before encoding
df_orig = df[['APR DRG Code', 'Year', 'Discharges']].copy()

# Concatenate the original DataFrame with the encoded features
df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)

print("\nStep 3: Preparing Data for Modeling")

# Fix infinity and extremely large values before model training
print("Checking for infinity and extreme values...")

# Only check numeric columns for infinities
numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64']).columns
inf_numeric_cols = [col for col in numeric_cols if np.isinf(df[col]).any()]
print(f"Columns with infinity values: {inf_numeric_cols}")

# Replace infinity values with NaN, then with the column median
for col in numeric_cols:
    # Replace inf with NaN
    mask = np.isinf(df[col])
    if mask.any():
        print(f"Replacing {mask.sum()} infinity values in {col}")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Replace NaN with median of column
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        
    # Cap extreme values at 99th percentile
    if (df[col].abs() > 1e6).any():
        print(f"Capping extreme values in {col}")
        upper_limit = df[col].quantile(0.99)
        lower_limit = df[col].quantile(0.01)
        df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)

# Determine the latest year in the dataset for temporal validation
max_year = df['Year'].max()
train_df = df[df['Year'] < max_year].copy()
test_df = df[df['Year'] == max_year].copy()

# Take a smaller sample for testing
sample_size = 10000  # Adjust as needed
train_df = train_df.sample(n=min(len(train_df), sample_size), random_state=42)
test_df = test_df.sample(n=min(len(test_df), sample_size), random_state=42)
print(f"Using reduced dataset: Train={len(train_df)}, Test={len(test_df)}")

print(f"Training data years: {train_df['Year'].min()} to {train_df['Year'].max()}")
print(f"Testing data year: {test_df['Year'].unique()}")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Function to prepare features and target for a specific model
def prepare_model_data(data, target_col, feature_cols=None):
    """
    Prepare features and target for modeling
    
    Parameters:
    -----------
    data : pandas DataFrame
        Input data
    target_col : str
        Column name of the target variable
    feature_cols : list, optional
        List of feature columns to use. If None, use all appropriate columns.
        
    Returns:
    --------
    X : pandas DataFrame
        Features
    y : pandas Series
        Target
    """
    # If feature columns not specified, use all appropriate columns
    if feature_cols is None:
        # Exclude target column, Facility Name, and the composite key
        exclude_cols = [target_col, 'Facility Name', 'Facility_DRG_Key', 'Year_Cat']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    # Extract features and target
    X = data[feature_cols].copy()
    y = data[target_col].copy()
    
    return X, y

print("\nTask 1: Predict Hospital & DRG Specific Metrics for Next Year")

# 1a. Predict Discharges
print("\nModel 1a: Predicting Discharges")
# Get all numeric features that don't include the target
exclude_cols = ['Discharges', 'Mean Cost', 'Median Cost', 'Mean Charge', 'Median Charge', 'Facility Name', 'Facility_DRG_Key', 'Year_Cat']
include_cols = [col for col in df.columns if col not in exclude_cols and col != 'Discharges']
X_train_1a, y_train_1a = prepare_model_data(train_df, 'Discharges', include_cols)
X_test_1a, y_test_1a = prepare_model_data(test_df, 'Discharges', include_cols)

# Train Gradient Boosting model for Discharges
model_1a = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model_1a.fit(X_train_1a, y_train_1a)

# Make predictions
y_pred_1a = model_1a.predict(X_test_1a)

# Evaluate the model
mse_1a = mean_squared_error(y_test_1a, y_pred_1a)
rmse_1a = np.sqrt(mse_1a)
mae_1a = mean_absolute_error(y_test_1a, y_pred_1a)
r2_1a = r2_score(y_test_1a, y_pred_1a)

print(f"Discharges Prediction - RMSE: {rmse_1a:.2f}, MAE: {mae_1a:.2f}, R²: {r2_1a:.4f}")

# 1b. Predict Median Costs
print("\nModel 1b: Predicting Median Costs")
# Include Discharges as a feature for cost prediction
include_cols_1b = include_cols + ['Discharges']
X_train_1b, y_train_1b = prepare_model_data(train_df, 'Median Cost', include_cols_1b)
X_test_1b, y_test_1b = prepare_model_data(test_df, 'Median Cost', include_cols_1b)

# Train Gradient Boosting model for Median Costs
model_1b = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model_1b.fit(X_train_1b, y_train_1b)

# Make predictions
y_pred_1b = model_1b.predict(X_test_1b)

# Evaluate the model
mse_1b = mean_squared_error(y_test_1b, y_pred_1b)
rmse_1b = np.sqrt(mse_1b)
mae_1b = mean_absolute_error(y_test_1b, y_pred_1b)
r2_1b = r2_score(y_test_1b, y_pred_1b)

print(f"Median Costs Prediction - RMSE: {rmse_1b:.2f}, MAE: {mae_1b:.2f}, R²: {r2_1b:.4f}")

# 1c. Predict Median Charges
print("\nModel 1c: Predicting Median Charges")
# Include Discharges and Median Cost as features for charge prediction
include_cols_1c = include_cols + ['Discharges', 'Median Cost'] 
X_train_1c, y_train_1c = prepare_model_data(train_df, 'Median Charge', include_cols_1c)
X_test_1c, y_test_1c = prepare_model_data(test_df, 'Median Charge', include_cols_1c)

# Train Gradient Boosting model for Median Charges
model_1c = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model_1c.fit(X_train_1c, y_train_1c)

# Make predictions
y_pred_1c = model_1c.predict(X_test_1c)

# Evaluate the model
mse_1c = mean_squared_error(y_test_1c, y_pred_1c)
rmse_1c = np.sqrt(mse_1c)
mae_1c = mean_absolute_error(y_test_1c, y_pred_1c)
r2_1c = r2_score(y_test_1c, y_pred_1c)

print(f"Median Charges Prediction - RMSE: {rmse_1c:.2f}, MAE: {mae_1c:.2f}, R²: {r2_1c:.4f}")

print("\nTask 2: Predict Total Expected Discharges by DRG Type")

# Aggregate data by DRG Code and Year - using the original copy of the data
drg_yearly = df_orig.groupby(['APR DRG Code', 'Year'])['Discharges'].sum().reset_index()
drg_yearly.rename(columns={'Discharges': 'Total_Discharges'}, inplace=True)

# Create feature for previous year's discharges
drg_yearly = drg_yearly.sort_values(['APR DRG Code', 'Year'])
drg_yearly['Prev_Year_Discharges'] = drg_yearly.groupby('APR DRG Code')['Total_Discharges'].shift(1)
drg_yearly['YoY_Change'] = (drg_yearly['Total_Discharges'] - drg_yearly['Prev_Year_Discharges']) / drg_yearly['Prev_Year_Discharges']
drg_yearly.fillna(0, inplace=True)

# Split into train and test sets
drg_train = drg_yearly[drg_yearly['Year'] < max_year]
drg_test = drg_yearly[drg_yearly['Year'] == max_year]

# Prepare features
drg_features = ['Year', 'Prev_Year_Discharges', 'YoY_Change']
# We need to one-hot encode the APR DRG Code for this task
encoder_drg = OneHotEncoder(sparse_output=False, drop='first')
X_drg_encoded = encoder_drg.fit_transform(drg_yearly[['APR DRG Code']])
drg_feature_names = [f"DRG_{cat}" for cat in encoder_drg.categories_[0][1:]]
drg_encoded_df = pd.DataFrame(X_drg_encoded, columns=drg_feature_names, index=drg_yearly.index)
drg_yearly_full = pd.concat([drg_yearly, drg_encoded_df], axis=1)

# Update train and test sets
drg_train = drg_yearly_full[drg_yearly_full['Year'] < max_year]
drg_test = drg_yearly_full[drg_yearly_full['Year'] == max_year]

X_train_2 = drg_train[drg_features]
y_train_2 = drg_train['Total_Discharges']
X_test_2 = drg_test[drg_features]
y_test_2 = drg_test['Total_Discharges']

# Train model
model_2 = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model_2.fit(X_train_2, y_train_2)

# Make predictions
y_pred_2 = model_2.predict(X_test_2)

# Evaluate model
mse_2 = mean_squared_error(y_test_2, y_pred_2)
rmse_2 = np.sqrt(mse_2)
mae_2 = mean_absolute_error(y_test_2, y_pred_2)
r2_2 = r2_score(y_test_2, y_pred_2)

print(f"Total DRG Discharges Prediction - RMSE: {rmse_2:.2f}, MAE: {mae_2:.2f}, R²: {r2_2:.4f}")

print("\nTask 3: Predict Mean Cost per Discharge")

# Prepare features - exclude other cost/charge metrics
exclude_cols_3 = ['Mean Cost', 'Median Cost', 'Mean Charge', 'Median Charge', 'Facility Name', 'Facility_DRG_Key', 'Year_Cat']
include_cols_3 = [col for col in df.columns if col not in exclude_cols_3]
X_train_3, y_train_3 = prepare_model_data(train_df, 'Mean Cost', include_cols_3)
X_test_3, y_test_3 = prepare_model_data(test_df, 'Mean Cost', include_cols_3)

# Train model
model_3 = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model_3.fit(X_train_3, y_train_3)

# Make predictions
y_pred_3 = model_3.predict(X_test_3)

# Evaluate model
mse_3 = mean_squared_error(y_test_3, y_pred_3)
rmse_3 = np.sqrt(mse_3)
mae_3 = mean_absolute_error(y_test_3, y_pred_3)
r2_3 = r2_score(y_test_3, y_pred_3)

print(f"Mean Cost Prediction - RMSE: {rmse_3:.2f}, MAE: {mae_3:.2f}, R²: {r2_3:.4f}")

print("\nStep 4: Model Evaluation & Interpretation")

# Feature importance for all models
plt.figure(figsize=(12, 16))

plt.subplot(3, 1, 1)
feature_importance_1a = pd.DataFrame({
    'Feature': X_train_1a.columns,
    'Importance': model_1a.feature_importances_
}).sort_values('Importance', ascending=False).head(10)
sns.barplot(x='Importance', y='Feature', data=feature_importance_1a)
plt.title('Feature Importance for Discharges Model')

plt.subplot(3, 1, 2)
feature_importance_1b = pd.DataFrame({
    'Feature': X_train_1b.columns,
    'Importance': model_1b.feature_importances_
}).sort_values('Importance', ascending=False).head(10)
sns.barplot(x='Importance', y='Feature', data=feature_importance_1b)
plt.title('Feature Importance for Median Cost Model')

plt.subplot(3, 1, 3)
feature_importance_3 = pd.DataFrame({
    'Feature': X_train_3.columns,
    'Importance': model_3.feature_importances_
}).sort_values('Importance', ascending=False).head(10)
sns.barplot(x='Importance', y='Feature', data=feature_importance_3)
plt.title('Feature Importance for Mean Cost Model')

plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plots saved to 'feature_importance.png'")

# SHAP values for model explanation
print("\nCalculating SHAP values for model interpretation...")

try:
    # SHAP for Discharges model - using smaller sample
    print("Calculating SHAP values for discharges model...")
    # Check if DataFrame or numpy array
    if hasattr(X_test_1a, 'iloc'):
        X_test_sample = X_test_1a.iloc[:1000]  # Sample for SHAP analysis
        X_train_sample = X_train_1a.iloc[:1000]
    else:
        X_test_sample = X_test_1a[:1000]
        X_train_sample = X_train_1a[:1000]
    
    explainer_1a = shap.Explainer(model_1a, X_train_sample)
    shap_values_1a = explainer_1a(X_test_sample)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_1a, X_test_sample, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance for Discharges Model")
    plt.tight_layout()
    plt.savefig('shap_discharges.png')
    print("SHAP plot for discharges saved")
except Exception as e:
    print(f"Error in SHAP analysis for Discharges: {e}")
    print("Continuing with next steps...")

try:
    # SHAP for Mean Cost model
    print("Calculating SHAP values for mean cost model...")
    # Check if DataFrame or numpy array
    if hasattr(X_test_3, 'iloc'):
        X_test_sample = X_test_3.iloc[:1000]  # Sample for SHAP analysis
        X_train_sample = X_train_3.iloc[:1000]
    else:
        X_test_sample = X_test_3[:1000]
        X_train_sample = X_train_3[:1000]
    
    explainer_3 = shap.Explainer(model_3, X_train_sample)
    shap_values_3 = explainer_3(X_test_sample)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_3, X_test_sample, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance for Mean Cost Model")
    plt.tight_layout()
    plt.savefig('shap_mean_cost.png')
    print("SHAP plot for mean cost saved")
except Exception as e:
    print(f"Error in SHAP analysis for Mean Cost: {e}")
    print("Continuing with next steps...")

print("\nStep 5: Forecasting for Next Year")

# Identify the next year for prediction
next_year = max_year + 1
print(f"Forecasting for year: {next_year}")

# Create a function to prepare data for the next year prediction
def prepare_forecast_data(df, year, features):
    """
    Prepare data for forecasting the next year
    
    Parameters:
    -----------
    df : pandas DataFrame
        Original dataset
    year : int
        Current year to base predictions on
    features : list
        List of features required by the model
        
    Returns:
    --------
    forecast_df : pandas DataFrame
        Data prepared for forecasting
    """
    # Filter data for the current year
    current_year_data = df[df['Year'] == year].copy()
    
    # Update year to next year
    current_year_data['Year'] = year + 1
    
    # We'll need to update YoY features if we have them
    yoy_cols = [col for col in current_year_data.columns if 'YoY' in col]
    for col in yoy_cols:
        # For simplicity, we'll use the average of YoY changes from recent years
        avg_yoy = df[df['Year'] >= year - 2][col].mean()
        current_year_data[col] = avg_yoy
    
    # Return only the features needed for prediction
    return current_year_data[features]

# Task 1a: Predict Discharges for next year
X_test_1a_cols = X_test_1a.columns.tolist()
forecast_data_1a = prepare_forecast_data(df, max_year, X_test_1a_cols)
forecast_discharges = model_1a.predict(forecast_data_1a)

# Create a working copy of the test dataframe to update with predictions
forecast_df = test_df.copy()
forecast_df['Year'] = next_year
forecast_df['Discharges'] = model_1a.predict(X_test_1a)  # Updated with predicted discharges

# Task 1b: Predict Median Costs for next year
X_test_1b_cols = X_test_1b.columns.tolist()
forecast_data_1b = prepare_forecast_data(df, max_year, X_test_1b_cols)
forecast_data_1b['Discharges'] = forecast_discharges  # Use the predicted discharges
forecast_median_costs = model_1b.predict(forecast_data_1b)

# Task 1c: Predict Median Charges for next year
X_test_1c_cols = X_test_1c.columns.tolist()
forecast_data_1c = prepare_forecast_data(df, max_year, X_test_1c_cols)
forecast_data_1c['Discharges'] = forecast_discharges  # Use the predicted discharges
forecast_data_1c['Median Cost'] = forecast_median_costs  # Use the predicted median costs
forecast_median_charges = model_1c.predict(forecast_data_1c)

# Task 2: Predict total DRG discharges for next year
# Update the year in the test data
drg_test_next_year = drg_test.copy()
drg_test_next_year['Year'] = next_year
drg_test_next_year['Prev_Year_Discharges'] = drg_test['Total_Discharges']  # Use the current year's data as previous
forecast_data_2 = drg_test_next_year[X_test_2.columns]
forecast_total_drg = model_2.predict(forecast_data_2)

# Task 3: Predict Mean Cost per Discharge for next year
X_test_3_cols = X_test_3.columns.tolist()
forecast_data_3 = prepare_forecast_data(df, max_year, X_test_3_cols)
forecast_data_3['Discharges'] = forecast_discharges  # Use the predicted discharges
forecast_mean_costs = model_3.predict(forecast_data_3)

print("\nStep 6: Documentation & Results")

# Extract necessary information for the forecast results
# We need to get back the original categorical values from the one-hot encoded data
# This is simplified for demonstration - in a real scenario we would map back to original values

# Prepare results data frames for Task 1
forecast_results_1 = pd.DataFrame({
    'Year': [next_year] * len(forecast_data_1a),
    'Predicted_Discharges': forecast_discharges,
    'Predicted_Median_Cost': forecast_median_costs,
    'Predicted_Median_Charge': forecast_median_charges
})

# Add some identifier columns (these will just be index values for simplicity)
forecast_results_1['Record_ID'] = range(len(forecast_results_1))

# Prepare results data frames for Task 2
forecast_results_2 = pd.DataFrame({
    'Year': [next_year] * len(forecast_data_2),
    'Record_ID': drg_test['APR DRG Code'].values,  # Using the original DRG Code
    'Predicted_Total_Discharges': forecast_total_drg
})

# Prepare results data frames for Task 3
forecast_results_3 = pd.DataFrame({
    'Year': [next_year] * len(forecast_data_3),
    'Record_ID': range(len(forecast_data_3)),
    'Predicted_Mean_Cost': forecast_mean_costs
})

# Save results to CSV files
forecast_results_1.to_csv('forecast_hospital_drg_metrics.csv', index=False)
forecast_results_2.to_csv('forecast_total_drg_discharges.csv', index=False)
forecast_results_3.to_csv('forecast_mean_costs.csv', index=False)

print("Forecast results saved to CSV files:")
print("- forecast_hospital_drg_metrics.csv")
print("- forecast_total_drg_discharges.csv")
print("- forecast_mean_costs.csv")

# Visualize predictions vs actual values
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.scatter(y_test_1a, y_pred_1a, alpha=0.5)
plt.plot([y_test_1a.min(), y_test_1a.max()], [y_test_1a.min(), y_test_1a.max()], 'r--')
plt.xlabel('Actual Discharges')
plt.ylabel('Predicted Discharges')
plt.title('Discharges: Actual vs Predicted')

plt.subplot(2, 2, 2)
plt.scatter(y_test_1b, y_pred_1b, alpha=0.5)
plt.plot([y_test_1b.min(), y_test_1b.max()], [y_test_1b.min(), y_test_1b.max()], 'r--')
plt.xlabel('Actual Median Cost')
plt.ylabel('Predicted Median Cost')
plt.title('Median Cost: Actual vs Predicted')

plt.subplot(2, 2, 3)
plt.scatter(y_test_2, y_pred_2, alpha=0.5)
plt.plot([y_test_2.min(), y_test_2.max()], [y_test_2.min(), y_test_2.max()], 'r--')
plt.xlabel('Actual Total DRG Discharges')
plt.ylabel('Predicted Total DRG Discharges')
plt.title('Total DRG Discharges: Actual vs Predicted')

plt.subplot(2, 2, 4)
plt.scatter(y_test_3, y_pred_3, alpha=0.5)
plt.plot([y_test_3.min(), y_test_3.max()], [y_test_3.min(), y_test_3.max()], 'r--')
plt.xlabel('Actual Mean Cost')
plt.ylabel('Predicted Mean Cost')
plt.title('Mean Cost: Actual vs Predicted')

plt.tight_layout()
plt.savefig('predictions_vs_actual.png')
print("Prediction vs actual plots saved to 'predictions_vs_actual.png'")

print("\nSummary of model performance:")
print(f"Task 1a - Discharges Prediction: R² = {r2_1a:.4f}, RMSE = {rmse_1a:.2f}")
print(f"Task 1b - Median Costs Prediction: R² = {r2_1b:.4f}, RMSE = {rmse_1b:.2f}")
print(f"Task 1c - Median Charges Prediction: R² = {r2_1c:.4f}, RMSE = {rmse_1c:.2f}")
print(f"Task 2 - Total DRG Discharges Prediction: R² = {r2_2:.4f}, RMSE = {rmse_2:.2f}")
print(f"Task 3 - Mean Cost Prediction: R² = {r2_3:.4f}, RMSE = {rmse_3:.2f}")

print("\nAnalysis completed successfully!") 