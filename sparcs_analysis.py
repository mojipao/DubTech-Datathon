#!/usr/bin/env python
# SPARCS Dataset Predictive Analysis

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
from scipy import sparse
from sklearn.inspection import permutation_importance

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
# Create separate encoders for each categorical column to avoid creating one huge matrix
encoded_dfs = []
encoders = {}
for col in categorical_cols:
    # Use sparse matrices to save memory
    encoder = OneHotEncoder(sparse_output=True, drop='first', handle_unknown='ignore')
    encoded = encoder.fit_transform(df[[col]])
    
    # Create feature names
    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
    
    # Keep track of the encoder for future predictions
    encoders[col] = encoder
    
    # Instead of creating a DataFrame, keep as sparse matrix with feature names
    encoded_dfs.append((encoded, feature_names))

# Save original columns needed for task 2 before encoding
df_orig = df[['APR DRG Code', 'Year', 'Discharges']].copy()

# Create a function to transform data for model input to ensure sparse matrices are preserved
def prepare_sparse_input(dataframe, feature_cols, categorical_encodings=None, encoders=None):
    """
    Prepare sparse input for model training/prediction that preserves sparsity
    
    Parameters:
    -----------
    dataframe : pandas DataFrame
        Input data frame with original columns
    feature_cols : list
        List of non-encoded feature columns to use
    categorical_encodings : list of tuples (sparse_matrix, feature_names), optional
        Pre-encoded categorical features as sparse matrices (for test/validation data)
    encoders : dict, optional
        Fitted encoders to transform categorical data (needed for training data)
        
    Returns:
    --------
    X : scipy.sparse.csr_matrix
        Sparse feature matrix suitable for training/prediction
    feature_names : list
        Names of all features in the same order as X
    """
    # Start with just the numeric features
    X_numeric = dataframe[feature_cols].copy()
    all_feature_names = feature_cols.copy()
    
    # Convert to sparse matrix
    X_sparse = sparse.csr_matrix(X_numeric.values)
    
    # If we have categorical encodings, add them
    matrices_to_combine = [X_sparse]
    
    if encoders:
        # We need to transform the categorical features for this specific dataframe
        for col, encoder in encoders.items():
            if col in dataframe.columns:
                encoded = encoder.transform(dataframe[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                matrices_to_combine.append(encoded)
                all_feature_names.extend(feature_names)
    
    # Combine all matrices horizontally
    X_combined = sparse.hstack(matrices_to_combine, format='csr')
    return X_combined, all_feature_names

# Drop original categorical columns
df = df.drop(categorical_cols, axis=1)

print("\nStep 3: Preparing Data for Modeling")

print("Enhanced Anomaly Detection and Handling")

# 1. Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"Found {duplicate_count} duplicate rows")
if duplicate_count > 0:
    print("Removing duplicate rows...")
    df = df.drop_duplicates()

# 2. Validate logical relationships (costs vs charges)
if all(col in df.columns for col in ['Mean Cost', 'Mean Charge']):
    illogical_records = df[df['Mean Cost'] > df['Mean Charge']].shape[0]
    print(f"Found {illogical_records} records where Mean Cost > Mean Charge (illogical)")
    if illogical_records > 0:
        # Flag these records
        df['Illogical_Cost_Charge'] = (df['Mean Cost'] > df['Mean Charge']).astype(int)
        # For severe cases, we could correct
        severe_cases = df[(df['Mean Cost'] > df['Mean Charge']*1.5)]
        print(f"  Of these, {severe_cases.shape[0]} are severe (cost > 1.5x charge)")
        if not severe_cases.empty:
            # For severe cases, swap the values as they were likely entered in wrong fields
            swap_indices = severe_cases.index
            print(f"  Swapping cost and charge values for {len(swap_indices)} severe cases")
            df.loc[swap_indices, ['Mean Cost', 'Mean Charge']] = df.loc[swap_indices, ['Mean Charge', 'Mean Cost']].values

# 3. Detect temporal disruptions (unusual YoY changes)
for metric in ['Discharges', 'Mean Cost', 'Median Cost']:
    if f'YoY_{metric}_Change' in df.columns:
        # Define threshold for unusual changes (e.g., >100% increase or >50% decrease)
        upper_threshold = 1.0  # 100% increase
        lower_threshold = -0.5  # 50% decrease
        unusual_increases = (df[f'YoY_{metric}_Change'] > upper_threshold).sum()
        unusual_decreases = (df[f'YoY_{metric}_Change'] < lower_threshold).sum()
        print(f"Unusual YoY changes in {metric}: {unusual_increases} large increases, {unusual_decreases} large decreases")
        
        # Flag unusual patterns
        df[f'Unusual_{metric}_Change'] = 0
        df.loc[df[f'YoY_{metric}_Change'] > upper_threshold, f'Unusual_{metric}_Change'] = 1
        df.loc[df[f'YoY_{metric}_Change'] < lower_threshold, f'Unusual_{metric}_Change'] = -1
        
        # Winsorize extremely unusual changes to reasonable bounds
        extreme_upper = df[f'YoY_{metric}_Change'].quantile(0.99)
        extreme_lower = df[f'YoY_{metric}_Change'].quantile(0.01)
        df[f'YoY_{metric}_Change'] = df[f'YoY_{metric}_Change'].clip(lower=extreme_lower, upper=extreme_upper)

# 4. Handle low outliers (negative values that shouldn't be negative)
for col in ['Discharges', 'Mean Cost', 'Median Cost', 'Mean Charge', 'Median Charge']:
    if col in df.columns:
        neg_values = (df[col] < 0).sum()
        if neg_values > 0:
            print(f"Found {neg_values} negative values in {col}")
            # For metrics that should never be negative, replace with 0
            df[col] = df[col].clip(lower=0)

# 5. Detect coding changes in DRG codes
if 'APR DRG Code' in df.columns and 'Year' in df.columns:
    # Check for DRG codes that appear or disappear abruptly
    drg_by_year = df.groupby(['Year'])['APR DRG Code'].nunique()
    print("Number of unique DRG codes by year:")
    print(drg_by_year)
    
    # Detect years with unusual changes in DRG code counts (possible coding system changes)
    drg_count_changes = drg_by_year.pct_change()
    unusual_years = drg_count_changes[abs(drg_count_changes) > 0.1].index.tolist()
    if unusual_years:
        print(f"Possible DRG coding system changes in years: {unusual_years}")
        # Create a feature to indicate pre/post coding system change
        latest_change_year = max(unusual_years)
        df['Post_DRG_Change'] = (df['Year'] > latest_change_year).astype(int)

# 6. Check for multimodal distributions
# Use a simple approach - check if variance is unexpectedly high, indicating possible multimodality
for col in ['Mean Cost', 'Median Cost', 'Mean Charge', 'Median Charge']:
    if col in df.columns:
        # Calculate coefficient of variation (normalized measure of dispersion)
        cv = df[col].std() / df[col].mean()
        print(f"Coefficient of variation for {col}: {cv:.2f}")
        if cv > 1.0:  # Rule of thumb for high variability
            print(f"  High variability detected in {col}, possible multimodal distribution")
            # Create binned version of this feature to handle multimodality
            df[f'{col}_Binned'] = pd.qcut(df[col], q=5, duplicates='drop', labels=False)
            print(f"  Created binned feature: {col}_Binned")

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
sample_size = 100000  # Reduced from 500,000 to manage memory better
train_df = train_df.sample(n=min(len(train_df), sample_size), random_state=42)
test_df = test_df.sample(n=min(len(test_df), sample_size), random_state=42)
print(f"Using reduced dataset: Train={len(train_df)}, Test={len(test_df)}")

print(f"Training data years: {train_df['Year'].min()} to {train_df['Year'].max()}")
print(f"Testing data year: {test_df['Year'].unique()}")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Function to prepare features and target for a specific model
def prepare_model_data(data, target_col, feature_cols=None, encoders_dict=None):
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
    encoders_dict : dict, optional
        Dictionary of fitted encoders for categorical features
        
    Returns:
    --------
    X : scipy.sparse.csr_matrix
        Features as sparse matrix
    y : pandas Series
        Target
    feature_names : list
        Names of all features in order
    """
    # If feature columns not specified, use all appropriate columns
    if feature_cols is None:
        # Exclude target column, Facility Name, and the composite key
        exclude_cols = [target_col, 'Facility Name', 'Facility_DRG_Key', 'Year_Cat']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    # Extract target
    y = data[target_col].copy()
    
    # Prepare sparse input features, using encoders to transform this specific dataframe
    X, feature_names = prepare_sparse_input(data, feature_cols, encoders=encoders_dict)
    
    return X, y, feature_names

print("\nTask 1: Predict Hospital & DRG Specific Metrics for Next Year")

# 1a. Predict Discharges
print("\nModel 1a: Predicting Discharges")
# Get all numeric features that don't include the target
exclude_cols = ['Discharges', 'Mean Cost', 'Median Cost', 'Mean Charge', 'Median Charge', 'Facility Name', 'Facility_DRG_Key', 'Year_Cat']
include_cols = [col for col in df.columns if col not in exclude_cols and col != 'Discharges']
X_train_1a, y_train_1a, feature_names_1a = prepare_model_data(train_df, 'Discharges', include_cols, encoders)
X_test_1a, y_test_1a, _ = prepare_model_data(test_df, 'Discharges', include_cols, encoders)

# Convert to dense arrays for HistGradientBoostingRegressor which doesn't accept sparse inputs
print("Converting sparse matrices to dense arrays...")
X_train_1a_dense = X_train_1a.toarray()
X_test_1a_dense = X_test_1a.toarray()

# Train Gradient Boosting model for Discharges
model_1a = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    max_depth=5, 
    random_state=42,
    categorical_features=None,  # Auto-detection for sparse matrices
)
model_1a.fit(X_train_1a_dense, y_train_1a)

# Make predictions
y_pred_1a = model_1a.predict(X_test_1a_dense)

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
X_train_1b, y_train_1b, feature_names_1b = prepare_model_data(train_df, 'Median Cost', include_cols_1b, encoders)
X_test_1b, y_test_1b, _ = prepare_model_data(test_df, 'Median Cost', include_cols_1b, encoders)

# Convert to dense arrays
X_train_1b_dense = X_train_1b.toarray()
X_test_1b_dense = X_test_1b.toarray()

# Train Gradient Boosting model for Median Costs
model_1b = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    categorical_features=None,  # Auto-detection for sparse matrices
)
model_1b.fit(X_train_1b_dense, y_train_1b)

# Make predictions
y_pred_1b = model_1b.predict(X_test_1b_dense)

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
X_train_1c, y_train_1c, feature_names_1c = prepare_model_data(train_df, 'Median Charge', include_cols_1c, encoders)
X_test_1c, y_test_1c, _ = prepare_model_data(test_df, 'Median Charge', include_cols_1c, encoders)

# Convert to dense arrays
X_train_1c_dense = X_train_1c.toarray()
X_test_1c_dense = X_test_1c.toarray()

# Train Gradient Boosting model for Median Charges
model_1c = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    categorical_features=None,  # Auto-detection for sparse matrices
)
model_1c.fit(X_train_1c_dense, y_train_1c)

# Make predictions
y_pred_1c = model_1c.predict(X_test_1c_dense)

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
# Only fill NaN values in numeric columns, not categorical
drg_yearly['Prev_Year_Discharges'] = drg_yearly['Prev_Year_Discharges'].fillna(0)
drg_yearly['YoY_Change'] = drg_yearly['YoY_Change'].fillna(0)

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
model_2 = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    categorical_features=None,  # Auto-detection for sparse matrices
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
X_train_3, y_train_3, feature_names_3 = prepare_model_data(train_df, 'Mean Cost', include_cols_3, encoders)
X_test_3, y_test_3, _ = prepare_model_data(test_df, 'Mean Cost', include_cols_3, encoders)

# Convert to dense arrays
X_train_3_dense = X_train_3.toarray()
X_test_3_dense = X_test_3.toarray()

# Train model
model_3 = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    categorical_features=None,  # Auto-detection for sparse matrices
)
model_3.fit(X_train_3_dense, y_train_3)

# Make predictions
y_pred_3 = model_3.predict(X_test_3_dense)

# Evaluate model
mse_3 = mean_squared_error(y_test_3, y_pred_3)
rmse_3 = np.sqrt(mse_3)
mae_3 = mean_absolute_error(y_test_3, y_pred_3)
r2_3 = r2_score(y_test_3, y_pred_3)

print(f"Mean Cost Prediction - RMSE: {rmse_3:.2f}, MAE: {mae_3:.2f}, R²: {r2_3:.4f}")

print("\nStep 4: Model Evaluation & Interpretation")

# Feature importance for all models
plt.figure(figsize=(12, 8))

# Use simplified feature importance calculation - avoid permutation importance which is memory intensive
print("Calculating simplified feature importance...")

# Calculate basic feature importance for model 1a
importance_1a = np.zeros(len(feature_names_1a))
# Extract feature importance if available, otherwise leave zeros
if hasattr(model_1a, 'feature_importances_'):
    importance_1a = model_1a.feature_importances_
# Get top 10 features by importance
top_idx_1a = np.argsort(importance_1a)[-10:]
feature_importance_1a = pd.DataFrame({
    'Feature': [feature_names_1a[i] for i in top_idx_1a],
    'Importance': importance_1a[top_idx_1a]
})

plt.subplot(2, 1, 1)
sns.barplot(x='Importance', y='Feature', data=feature_importance_1a)
plt.title('Feature Importance for Discharges Model')

# Calculate basic feature importance for model 3
importance_3 = np.zeros(len(feature_names_3))
# Extract feature importance if available, otherwise leave zeros
if hasattr(model_3, 'feature_importances_'):
    importance_3 = model_3.feature_importances_
# Get top 10 features by importance
top_idx_3 = np.argsort(importance_3)[-10:]
feature_importance_3 = pd.DataFrame({
    'Feature': [feature_names_3[i] for i in top_idx_3],
    'Importance': importance_3[top_idx_3]
})

plt.subplot(2, 1, 2)
sns.barplot(x='Importance', y='Feature', data=feature_importance_3)
plt.title('Feature Importance for Mean Cost Model')

plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plots saved to 'feature_importance.png'")

print("\nPerforming optimized SHAP analysis on very small subset...")

# Helper function to clean up memory
def clean_memory():
    """Force garbage collection to free memory"""
    import gc
    gc.collect()

try:
    # SHAP for Discharges model - using extremely small sample
    print("Calculating SHAP values for discharges model (small subset)...")
    
    # Select a very small random sample (500 instances)
    tiny_sample_size = 500
    sample_idx = np.random.choice(X_test_1a_dense.shape[0], size=min(tiny_sample_size, X_test_1a_dense.shape[0]), replace=False)
    
    # Use the FULL feature set, not just top features
    X_test_tiny = X_test_1a_dense[sample_idx]
    
    # Create explainer with very small background set
    background = X_test_tiny[:100]  # Just 100 background samples
    explainer_1a = shap.Explainer(model_1a.predict, background)
    
    # Calculate SHAP values only on this tiny subset
    shap_values_1a = explainer_1a(X_test_tiny)
    
    # 6. Create plot - but focus on top 10 features only
    plt.figure(figsize=(10, 6))
    # Set feature names for all features
    shap_values_1a.feature_names = feature_names_1a
    # Show only top features in the plot
    shap.summary_plot(shap_values_1a, X_test_tiny, plot_type="bar", max_display=10, show=False)
    plt.title("SHAP Feature Importance for Discharges Model (Top Features)")
    plt.tight_layout()
    plt.savefig('shap_discharges.png')
    print("SHAP plot for discharges saved")
    
    # Clean up memory
    del X_test_tiny, shap_values_1a, explainer_1a, background
    clean_memory()
    
except Exception as e:
    print(f"Error in SHAP analysis for Discharges: {e}")
    print("Continuing with next steps...")

try:
    # SHAP for Mean Cost model
    print("Calculating SHAP values for mean cost model (small subset)...")
    
    # Select a very small random sample (500 instances)
    tiny_sample_size = 500
    sample_idx = np.random.choice(X_test_3_dense.shape[0], size=min(tiny_sample_size, X_test_3_dense.shape[0]), replace=False)
    
    # Use the FULL feature set, not just top features
    X_test_tiny = X_test_3_dense[sample_idx]
    
    # Create explainer with very small background set
    background = X_test_tiny[:100]  # Just 100 background samples
    explainer_3 = shap.Explainer(model_3.predict, background)
    
    # Calculate SHAP values only on this tiny subset
    shap_values_3 = explainer_3(X_test_tiny)
    
    # Create plot - but focus on top 10 features only
    plt.figure(figsize=(10, 6))
    # Set feature names for all features
    shap_values_3.feature_names = feature_names_3
    # Show only top features in the plot
    shap.summary_plot(shap_values_3, X_test_tiny, plot_type="bar", max_display=10, show=False)
    plt.title("SHAP Feature Importance for Mean Cost Model (Top Features)")
    plt.tight_layout()
    plt.savefig('shap_mean_cost.png')
    print("SHAP plot for mean cost saved")
    
    # Clean up memory
    del X_test_tiny, shap_values_3, explainer_3, background
    clean_memory()
    
except Exception as e:
    print(f"Error in SHAP analysis for Mean Cost: {e}")
    print("Continuing with next steps...")

print("\nStep 5: Forecasting for Next Year")

# Create a function to prepare data for the next year prediction
def prepare_forecast_data(df, year, feature_cols, encoders_dict):
    """
    Prepare data for forecasting the next year
    
    Parameters:
    -----------
    df : pandas DataFrame
        Original dataset
    year : int
        Current year to base predictions on
    feature_cols : list
        List of feature columns to use
    encoders_dict : dict
        Dictionary of encoders for categorical features
        
    Returns:
    --------
    forecast_df : numpy.ndarray
        Dense array prepared for forecasting
    """
    # Filter data for the current year
    current_year_data = df[df['Year'] == year].copy()
    
    # Update year to next year
    current_year_data['Year'] = year + 1
    
    # We'll need to update YoY features if we have them
    yoy_cols = [col for col in feature_cols if 'YoY' in col and col in current_year_data.columns]
    for col in yoy_cols:
        # For simplicity, we'll use the average of YoY changes from recent years
        avg_yoy = df[df['Year'] >= year - 2][col].mean()
        current_year_data[col] = avg_yoy
    
    # Only include the exact feature columns used for training
    current_year_data_subset = current_year_data[feature_cols].copy()
    
    # Process with our data preparation functions and convert to dense
    X, _ = prepare_sparse_input(current_year_data_subset, feature_cols, encoders=encoders_dict)
    return X.toarray()

# Identify the next year for prediction
next_year = max_year + 1
print(f"Forecasting for year: {next_year}")

# Task 1a: Predict Discharges for next year
# Get the exact feature columns used for model 1a
model_1a_feature_cols = include_cols.copy()
forecast_data_1a = prepare_forecast_data(df, max_year, model_1a_feature_cols, encoders)
# Verify dimensions match training data
print(f"Model 1a: forecast_data has {forecast_data_1a.shape[1]} features, model expects {X_train_1a_dense.shape[1]} features")
forecast_discharges = model_1a.predict(forecast_data_1a)

# Create a working copy of the test dataframe to update with predictions
forecast_df = test_df.copy()
forecast_df['Year'] = next_year
forecast_df['Discharges'] = model_1a.predict(X_test_1a_dense)  # Updated with predicted discharges

# Task 1b: Predict Median Costs for next year
# Get the exact feature columns used for model 1b
model_1b_feature_cols = include_cols_1b.copy()
forecast_data_1b = prepare_forecast_data(df, max_year, model_1b_feature_cols, encoders)
# Verify dimensions match training data
print(f"Model 1b: forecast_data has {forecast_data_1b.shape[1]} features, model expects {X_train_1b_dense.shape[1]} features")
forecast_median_costs = model_1b.predict(forecast_data_1b)

# Task 1c: Predict Median Charges for next year
# Get the exact feature columns used for model 1c
model_1c_feature_cols = include_cols_1c.copy()
forecast_data_1c = prepare_forecast_data(df, max_year, model_1c_feature_cols, encoders)
# Verify dimensions match training data
print(f"Model 1c: forecast_data has {forecast_data_1c.shape[1]} features, model expects {X_train_1c_dense.shape[1]} features")
forecast_median_charges = model_1c.predict(forecast_data_1c)

# Task 2: Predict total DRG discharges for next year
# Update the year in the test data
drg_test_next_year = drg_test.copy()
drg_test_next_year['Year'] = next_year
drg_test_next_year['Prev_Year_Discharges'] = drg_test['Total_Discharges']  # Use the current year's data as previous
forecast_data_2 = drg_test_next_year[X_test_2.columns]
forecast_total_drg = model_2.predict(forecast_data_2)

# Task 3: Predict Mean Cost per Discharge for next year
# Get the exact feature columns used for model 3
model_3_feature_cols = include_cols_3.copy()
forecast_data_3 = prepare_forecast_data(df, max_year, model_3_feature_cols, encoders)
# Verify dimensions match training data
print(f"Model 3: forecast_data has {forecast_data_3.shape[1]} features, model expects {X_train_3_dense.shape[1]} features")
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