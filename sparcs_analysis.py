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
import warnings
import gc
import time
from datetime import datetime

# Try to import LightGBM, but make it optional
USE_LIGHTGBM = False  # Default to False
print("LightGBM benchmarking disabled by default.")

# Suppress warnings
warnings.filterwarnings('ignore')

# Global verbose flag to control print outputs
VERBOSE = True

# Function to print if verbose is enabled
def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# Set random seed for reproducibility
np.random.seed(42)

# Display settings
pd.set_option('display.max_columns', None)
sns.set(style='whitegrid')

# Start timing
start_time = time.time()

print("Step 1: Data Loading & Preparation")
# Load the dataset
file_path = 'Hospital_Inpatient_Discharges__SPARCS_De-Identified___Cost_Transparency__Beginning_2009_20250426.csv'
df = pd.read_csv(file_path)

# Display basic information
vprint(f"Dataset shape: {df.shape}")
vprint(df.head())

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
vprint("Filtered dataset:")
vprint(df.head())

# Check for missing values
missing_values = df.isnull().sum()
vprint("Missing values per column:")
vprint(missing_values)

# Check data types
vprint("\nData types:")
vprint(df.dtypes)

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
        # Fill NaN values with 0 to avoid issues later
        df[col] = df[col].fillna(0).astype(np.float64)

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

# Enhanced Feature Engineering: Add facility regional clustering
# Extract facility information if available
if 'Facility Name' in df.columns:
    # Extract region from facility name if possible (looking for regional indicators)
    df['Facility_Region'] = df['Facility Name'].astype(str).str.extract(r'(NORTH|SOUTH|EAST|WEST|CENTRAL|UPSTATE|DOWNSTATE|NYC|BROOKLYN|BRONX|MANHATTAN|QUEENS|STATEN ISLAND)', expand=False)
    df['Facility_Region'] = df['Facility_Region'].fillna('OTHER')
    df['Facility_Region'] = df['Facility_Region'].astype('category')
    
    # Create aggregate stats by region
    region_aggs = df.groupby('Facility_Region').agg({
        'Discharges': ['mean', 'median', 'std']
    }).reset_index()
    
    region_aggs.columns = ['Facility_Region', 'Region_Avg_Discharges', 'Region_Median_Discharges', 'Region_Std_Discharges']
    df = pd.merge(df, region_aggs, on='Facility_Region', how='left')

# Enhanced Feature Engineering: Add temporal features
# Create more sophisticated Year-over-Year (YoY) trends
print("Adding enhanced temporal features...")

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

# Add lagged features (previous 1 and 2 years)
df['Prev_Year_Discharges'] = df.groupby('Facility_DRG_Key')['Discharges'].shift(1)
df['Prev2_Year_Discharges'] = df.groupby('Facility_DRG_Key')['Discharges'].shift(2)
df['Prev_Year_Mean_Cost'] = df.groupby('Facility_DRG_Key')['Mean Cost'].shift(1)
df['Prev_Year_Median_Cost'] = df.groupby('Facility_DRG_Key')['Median Cost'].shift(1)

# Add trend indicators: is the facility/DRG growing or shrinking?
df['Discharge_Trend'] = df.groupby('Facility_DRG_Key')['Discharges'].rolling(window=2, min_periods=2).apply(lambda x: (x.iloc[1] - x.iloc[0])/x.iloc[0] if x.iloc[0] != 0 else 0).reset_index(level=0, drop=True)
df['Discharge_Trend_Direction'] = np.sign(df['Discharge_Trend'])

# Add special indication for facilities with consistent growth/decline
df['Consistent_Growth'] = 0
df.loc[(df['YoY_Discharges_Change'] > 0) & (df['Discharge_Trend'] > 0), 'Consistent_Growth'] = 1
df.loc[(df['YoY_Discharges_Change'] < 0) & (df['Discharge_Trend'] < 0), 'Consistent_Growth'] = -1

# Enhanced Feature Engineering: Add health event indicators
print("Adding health event indicators...")

# Add COVID period indicator (2020-2022)
df['COVID_Period'] = 0
df.loc[df['Year'].between(2020, 2022), 'COVID_Period'] = 1

# Add seasonal indicators if month data is available
if 'Month' in df.columns:
    # Flu season indicator (roughly October through March)
    df['Flu_Season'] = 0
    df.loc[df['Month'].isin([1, 2, 3, 10, 11, 12]), 'Flu_Season'] = 1
else:
    # If no month data, create a proxy based on the DRG code for respiratory conditions
    respiratory_conditions = ['4', '7', '13', '14'] # Example DRG codes - replace with actual respiratory DRGs
    df['Respiratory_DRG'] = 0
    df.loc[df['APR DRG Code'].astype(str).str.startswith(tuple(respiratory_conditions)), 'Respiratory_DRG'] = 1

# Add interaction terms between relevant features
df['COVID_Respiratory_Impact'] = df.get('COVID_Period', 0) * df.get('Respiratory_DRG', 0)
df['Prev_Year_Impact'] = df['Prev_Year_Discharges'] * df.get('COVID_Period', 0)

# Fill NaN values in YoY changes and lagged features (first year will have NaN)
lag_cols = ['Prev_Year_Discharges', 'Prev2_Year_Discharges', 'Prev_Year_Mean_Cost', 
            'Prev_Year_Median_Cost', 'Discharge_Trend']
df[lag_cols] = df[lag_cols].fillna(0).astype(np.float64)
yoy_cols = [col for col in df.columns if 'YoY' in col]
df[yoy_cols] = df[yoy_cols].fillna(0).astype(np.float64)

# Replace infinity values that could cause issues
for col in yoy_cols:
    df[col] = df[col].replace([np.inf, -np.inf], 0)

# Merge the aggregated features back to the original dataframe
df = pd.merge(df, facility_aggs, on='Facility Id', how='left')
df = pd.merge(df, drg_aggs, on='APR DRG Code', how='left')
df = pd.merge(df, severity_aggs, on='APR Severity of Illness Code', how='left')

# Create temporal indicators - year as a categorical feature
df['Year_Cat'] = df['Year'].astype('category')

# One-hot encode categorical variables
print("\nEncoding categorical variables...")
# Create separate encoders for each categorical column to avoid creating one huge matrix
encoded_dfs = []
encoders = {}
for col in categorical_cols:
    try:
        # Convert categorical variables to string type to prevent encoding issues
        # Use sparse matrices to save memory
        encoder = OneHotEncoder(sparse_output=True, drop='first', handle_unknown='ignore', dtype=np.float64)
        # Ensure input is properly formatted as strings
        input_data = df[[col]].astype(str)
        encoded = encoder.fit_transform(input_data)
        
        # Create feature names
        feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
        
        # Keep track of the encoder for future predictions
        encoders[col] = encoder
        
        # Instead of creating a DataFrame, keep as sparse matrix with feature names
        encoded_dfs.append((encoded, feature_names))
    except Exception as e:
        print(f"Error encoding categorical column {col}: {e}")
        continue

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
    
    # Ensure all numeric columns have proper data types before conversion
    for col in X_numeric.columns:
        # Force conversion to float64 to ensure compatibility with sparse matrix
        X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce').fillna(0).astype(np.float64)
    
    # Convert to sparse matrix
    X_sparse = sparse.csr_matrix(X_numeric.values, dtype=np.float64)
    
    # If we have categorical encodings, add them
    matrices_to_combine = [X_sparse]
    
    if encoders:
        # We need to transform the categorical features for this specific dataframe
        for col, encoder in encoders.items():
            if col in dataframe.columns:
                try:
                    # Ensure the input to the encoder is properly formatted
                    input_data = dataframe[[col]].astype(str)
                    encoded = encoder.transform(input_data)
                    
                    # Ensure encoded data is float64 for sparse matrix compatibility
                    if not isinstance(encoded, sparse.csr_matrix):
                        encoded = sparse.csr_matrix(encoded, dtype=np.float64)
                    elif encoded.dtype != np.float64:
                        # Convert existing sparse matrix to float64
                        encoded = encoded.astype(np.float64)
                    
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                    matrices_to_combine.append(encoded)
                    all_feature_names.extend(feature_names)
                except Exception as e:
                    print(f"Error encoding {col}: {e}")
                    # Skip this encoder if there's an error
                    continue
    
    # Combine all matrices horizontally - ensure all have same data type
    try:
        X_combined = sparse.hstack(matrices_to_combine, format='csr', dtype=np.float64)
        return X_combined, all_feature_names
    except ValueError as e:
        print(f"Error combining matrices: {e}")
        # For debugging, print data types of all matrices
        for i, mat in enumerate(matrices_to_combine):
            print(f"Matrix {i} dtype: {mat.dtype}, shape: {mat.shape}")
        # Return just the numeric part if we can't combine
        return X_sparse, all_feature_names[:X_sparse.shape[1]]

# Drop original categorical columns
df = df.drop(categorical_cols, axis=1)

print("\nStep 3: Preparing Data for Modeling")

print("Enhanced Anomaly Detection and Handling")

# 1. Check for duplicates
duplicate_count = df.duplicated().sum()
vprint(f"Found {duplicate_count} duplicate rows")
if duplicate_count > 0:
    print("Removing duplicate rows...")
    df = df.drop_duplicates()

# 2. Validate logical relationships (costs vs charges)
if all(col in df.columns for col in ['Mean Cost', 'Mean Charge']):
    illogical_records = df[df['Mean Cost'] > df['Mean Charge']].shape[0]
    vprint(f"Found {illogical_records} records where Mean Cost > Mean Charge (illogical)")
    if illogical_records > 0:
        # Flag these records
        df['Illogical_Cost_Charge'] = (df['Mean Cost'] > df['Mean Charge']).astype(int)
        # For severe cases, we could correct
        severe_cases = df[(df['Mean Cost'] > df['Mean Charge']*1.5)]
        vprint(f"  Of these, {severe_cases.shape[0]} are severe (cost > 1.5x charge)")
        if not severe_cases.empty:
            # For severe cases, swap the values as they were likely entered in wrong fields
            swap_indices = severe_cases.index
            vprint(f"  Swapping cost and charge values for {len(swap_indices)} severe cases")
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

# Determine a better year for temporal validation (pre-COVID)
print("\nAdjusting temporal validation to avoid COVID period...")
available_years = sorted(df['Year'].unique())
print(f"Available years in dataset: {available_years}")

# Define all utility and task functions
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

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance and print metrics"""
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print evaluation metrics
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    return y_pred, rmse, mae, r2

def train_model(X_train, y_train, X_test, y_test, model_name, prevent_leakage_features=None, use_lightgbm=False):
    """Train model with feature leakage prevention and evaluation"""
    print(f"\nTraining {model_name} model...")
    
    # Remove features that could cause target leakage
    if prevent_leakage_features:
        leak_features_present = [f for f in prevent_leakage_features if f in X_train.columns]
        if leak_features_present:
            print(f"Removing potential leakage features: {leak_features_present}")
            X_train = X_train.drop(columns=leak_features_present)
            X_test = X_test.drop(columns=leak_features_present)
    
    # Convert to dense if not already
    if isinstance(X_train, sparse.csr_matrix):
        X_train_model = X_train.toarray()
        X_test_model = X_test.toarray()
    else:
        X_train_model = X_train
        X_test_model = X_test
        
    # Use LightGBM if requested and available, otherwise use HistGradientBoostingRegressor
    if use_lightgbm and USE_LIGHTGBM:
        model = lgbm.LGBMRegressor(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=200,
            reg_lambda=0.1,
            random_state=42
        )
    else:
        if use_lightgbm and not USE_LIGHTGBM:
            print("LightGBM was requested but is not available. Using HistGradientBoostingRegressor instead.")
            
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=8,
            l2_regularization=0.1,
    random_state=42,
            categorical_features=None,
        )
    
    # Train the model
    model.fit(X_train_model, y_train)
    
    # Evaluate model
    y_pred, rmse, mae, r2 = evaluate_model(model, X_test_model, y_test, model_name)
    
    return model, y_pred, rmse, mae, r2, X_train_model, X_test_model

def generate_shap_plots(model, X_test, feature_names, model_name, max_display=10, sample_size=300):
    """Generate SHAP summary and dependence plots for model interpretability"""
    print(f"\nGenerating SHAP plots for {model_name}...")
    
    try:
        # Make sure we're working with numpy arrays
        if isinstance(X_test, pd.DataFrame):
            feature_names = X_test.columns.tolist()
            X_sample = X_test.values
        else:
            X_sample = X_test
        
        # Use a reasonable sample size for SHAP analysis
        if len(X_sample) > sample_size:
            # Take a random sample of rows (not indices)
            sample_indices = np.random.choice(len(X_sample), size=min(sample_size, len(X_sample)), replace=False)
            X_sample = X_sample[sample_indices]
        
        # Create a smaller background dataset for efficiency
        background_size = min(100, len(X_sample))
        background = X_sample[:background_size]
        
        print(f"Running SHAP analysis on {len(X_sample)} samples with {len(feature_names)} features...")
        
        # Create SHAP explainer - use model.predict directly
        explainer = shap.Explainer(model.predict, background)
        
        # Calculate SHAP values
        shap_values = explainer(X_sample)
        
        # Create SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values=shap_values, 
            features=X_sample, 
            feature_names=feature_names,
            max_display=min(max_display, len(feature_names)),
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig(f'shap_summary_{model_name.lower().replace(" ", "_")}.png', dpi=300)
        plt.close()
        print(f"SHAP summary plot saved for {model_name}")
        
        # Find the top feature based on SHAP values
        if len(feature_names) > 0:
            # Calculate mean absolute SHAP values for each feature
            mean_abs_shap = np.abs(shap_values.values).mean(0)
            top_feature_idx = np.argmax(mean_abs_shap)
            top_feature = feature_names[top_feature_idx]
            
            # Generate a dependence plot for the top feature
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                ind=top_feature_idx, 
                shap_values=shap_values.values, 
                features=X_sample, 
                feature_names=feature_names,
                show=False
            )
            plt.title(f"SHAP Dependence Plot for {top_feature} in {model_name}")
            plt.tight_layout()
            plt.savefig(f'shap_dependence_{model_name.lower().replace(" ", "_")}.png', dpi=300)
            plt.close()
            print(f"SHAP dependence plot saved for {model_name}")
            
            return top_feature
        
        return None
            
    except Exception as e:
        print(f"Error in SHAP analysis for {model_name}: {str(e)}")
        
        # Fallback to model's feature_importances_ if available
        if hasattr(model, 'feature_importances_'):
            print(f"Using model's built-in feature importance for {model_name} instead")
            importances = model.feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10 features
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance for {model_name}')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
            plt.close()
            
            # Return the top feature
            top_idx = indices[-1]
            return feature_names[top_idx]
        
        return None
    finally:
        # Clean up memory
        gc.collect()

def train_and_interpret_model(X_train, y_train, X_test, y_test, model_name, 
                             feature_names, prevent_leakage_features=None, 
                             use_lightgbm=False):
    """Combined function to train, evaluate, and interpret a model"""
    # Train and evaluate the model
    model, y_pred, rmse, mae, r2, X_train_dense, X_test_dense = train_model(
        X_train, y_train, X_test, y_test, model_name, 
        prevent_leakage_features, use_lightgbm
    )
    
    # Save test data and predictions for later visualization
    globals()[f"y_test_{model_name.replace(' ', '_').lower()}"] = y_test
    globals()[f"y_pred_{model_name.replace(' ', '_').lower()}"] = y_pred
    
    # Generate simplified feature importance plots instead of SHAP
    top_feature = generate_shap_plots(model, X_test_dense, feature_names, model_name)
    
    # Print top feature if available
    if top_feature:
        print(f"Top feature for {model_name}: {top_feature}")
    
    return model, y_pred, rmse, mae, r2

def run_task1a_discharge_prediction(train_df, test_df, include_cols, encoders):
    """Task 1a: Predict Hospital & DRG Specific Metrics - Discharges"""
    print("\n" + "="*80)
    print("Task 1a: Predict Hospital & DRG Specific Metrics - Discharges")
    print("="*80)
    
    # Define features that could cause leakage for Discharges prediction
    leakage_features = ['Mean Cost', 'Median Cost', 'Mean Charge', 'Median Charge']
    
    # Prepare data for modeling
    X_train, y_train, feature_names = prepare_model_data(train_df, 'Discharges', include_cols, encoders)
    X_test, y_test, _ = prepare_model_data(test_df, 'Discharges', include_cols, encoders)
    
    # Convert sparse matrices to dataframes for leakage prevention
    X_train_df = pd.DataFrame(X_train.toarray(), columns=feature_names)
    X_test_df = pd.DataFrame(X_test.toarray(), columns=feature_names)
    
    # Train, evaluate, and interpret model
    model, y_pred, rmse, mae, r2 = train_and_interpret_model(
        X_train_df, y_train, X_test_df, y_test, 
        "Discharges Prediction",
        feature_names, 
        prevent_leakage_features=leakage_features
    )
    
    return model, y_pred, rmse, mae, r2, feature_names

def run_task1b_median_cost_prediction(train_df, test_df, include_cols, encoders):
    """Task 1b: Predict Hospital & DRG Specific Metrics - Median Costs"""
    print("\n" + "="*80)
    print("Task 1b: Predict Hospital & DRG Specific Metrics - Median Costs")
    print("="*80)
    
    # Define features that could cause leakage for Median Cost prediction
    leakage_features = ['Mean Cost', 'Mean Charge', 'Median Charge']
    
    # Include Discharges as a feature for cost prediction
    include_cols_1b = include_cols + ['Discharges']
    
    # Prepare data for modeling
    X_train, y_train, feature_names = prepare_model_data(train_df, 'Median Cost', include_cols_1b, encoders)
    X_test, y_test, _ = prepare_model_data(test_df, 'Median Cost', include_cols_1b, encoders)
    
    # Convert sparse matrices to dataframes for leakage prevention
    X_train_df = pd.DataFrame(X_train.toarray(), columns=feature_names)
    X_test_df = pd.DataFrame(X_test.toarray(), columns=feature_names)
    
    # Normal modeling approach
    model, y_pred, rmse, mae, r2 = train_and_interpret_model(
        X_train_df, y_train, X_test_df, y_test, 
        "Median Cost Prediction",
        feature_names, 
        prevent_leakage_features=leakage_features
    )
    
    # Optional: LightGBM benchmarking
    if USE_LIGHTGBM:
        print("\nBenchmarking with LightGBM...")
        lgbm_model, lgbm_y_pred, lgbm_rmse, lgbm_mae, lgbm_r2 = train_and_interpret_model(
            X_train_df, y_train, X_test_df, y_test, 
            "Median Cost LightGBM",
            feature_names, 
            prevent_leakage_features=leakage_features,
            use_lightgbm=True
        )
        
        # Compare models
        print("\nModel Comparison for Median Cost Prediction:")
        print(f"HistGradientBoosting - RMSE: {rmse:.2f}, R²: {r2:.4f}")
        print(f"LightGBM - RMSE: {lgbm_rmse:.2f}, R²: {lgbm_r2:.4f}")
    
    return model, y_pred, rmse, mae, r2, feature_names

def run_task1c_median_charge_prediction(train_df, test_df, include_cols, encoders):
    """Task 1c: Predict Hospital & DRG Specific Metrics - Median Charges"""
    print("\n" + "="*80)
    print("Task 1c: Predict Hospital & DRG Specific Metrics - Median Charges")
    print("="*80)
    
    # Define features that could cause leakage for Median Charge prediction
    leakage_features = ['Mean Cost', 'Mean Charge', 'Median Cost']
    
    # Include Discharges and (optionally) Median Cost as features
    include_cols_1c = include_cols + ['Discharges'] 
    
    # Prepare data for modeling
    X_train, y_train, feature_names = prepare_model_data(train_df, 'Median Charge', include_cols_1c, encoders)
    X_test, y_test, _ = prepare_model_data(test_df, 'Median Charge', include_cols_1c, encoders)
    
    # Convert sparse matrices to dataframes for leakage prevention
    X_train_df = pd.DataFrame(X_train.toarray(), columns=feature_names)
    X_test_df = pd.DataFrame(X_test.toarray(), columns=feature_names)
    
    # Train, evaluate, and interpret model
    model, y_pred, rmse, mae, r2 = train_and_interpret_model(
        X_train_df, y_train, X_test_df, y_test, 
        "Median Charge Prediction",
        feature_names, 
        prevent_leakage_features=leakage_features
    )
    
    return model, y_pred, rmse, mae, r2, feature_names

def run_task2_total_drg_prediction(df_orig, test_year):
    """Task 2: Predict Total Expected Discharges by DRG Type"""
    print("\n" + "="*80)
    print("Task 2: Predict Total Expected Discharges by DRG Type")
    print("="*80)
    
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
    drg_train = drg_yearly[drg_yearly['Year'] < test_year]
    drg_test = drg_yearly[drg_yearly['Year'] == test_year]
    
    # Prepare features
    drg_features = ['Year', 'Prev_Year_Discharges', 'YoY_Change']
    
    # One-hot encode the APR DRG Code for this task
    try:
        encoder_drg = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        X_drg_encoded = encoder_drg.fit_transform(drg_yearly[['APR DRG Code']].astype(str))
        
        # Create feature names for encoded features
        drg_feature_names = [f"DRG_{cat}" for cat in encoder_drg.categories_[0][1:]]
        
        # Convert encoded features to DataFrame
        drg_encoded_df = pd.DataFrame(X_drg_encoded, columns=drg_feature_names, index=drg_yearly.index)
        
        # Combine features with encoded DRG
        drg_yearly_full = pd.concat([drg_yearly, drg_encoded_df], axis=1)
        
        # Update train and test sets
        drg_train = drg_yearly_full[drg_yearly_full['Year'] < test_year]
        drg_test = drg_yearly_full[drg_yearly_full['Year'] == test_year]
    except Exception as e:
        print(f"Error in one-hot encoding DRG codes: {e}")
        print("Proceeding without DRG code encoding")
    
    # Prepare feature dataframes
    X_train = drg_train[drg_features]
    y_train = drg_train['Total_Discharges']
    X_test = drg_test[drg_features]
    y_test = drg_test['Total_Discharges']
    
    # No leakage features for this task
    # Train, evaluate, and interpret model
    model, y_pred, rmse, mae, r2 = train_and_interpret_model(
        X_train, y_train, X_test, y_test, 
        "Total DRG Discharges Prediction",
        drg_features
    )
    
    return model, y_pred, rmse, mae, r2, drg_test

def run_task3_mean_cost_prediction(train_df, test_df, include_cols, encoders):
    """Task 3: Predict Mean Cost per Discharge"""
    print("\n" + "="*80)
    print("Task 3: Predict Mean Cost per Discharge")
    print("="*80)
    
    # Define features that could cause leakage for Mean Cost prediction
    leakage_features = ['Median Cost', 'Mean Charge', 'Median Charge']
    
    # Prepare features - exclude other cost/charge metrics
    exclude_cols_3 = ['Mean Cost', 'Median Cost', 'Mean Charge', 'Median Charge', 'Facility Name', 'Facility_DRG_Key', 'Year_Cat']
    include_cols_3 = [col for col in train_df.columns if col not in exclude_cols_3]
    
    # Prepare data for modeling
    X_train, y_train, feature_names = prepare_model_data(train_df, 'Mean Cost', include_cols_3, encoders)
    X_test, y_test, _ = prepare_model_data(test_df, 'Mean Cost', include_cols_3, encoders)
    
    # Convert sparse matrices to dataframes for leakage prevention
    X_train_df = pd.DataFrame(X_train.toarray(), columns=feature_names)
    X_test_df = pd.DataFrame(X_test.toarray(), columns=feature_names)
    
    # Train, evaluate, and interpret model
    model, y_pred, rmse, mae, r2 = train_and_interpret_model(
        X_train_df, y_train, X_test_df, y_test, 
        "Mean Cost Prediction",
        feature_names, 
        prevent_leakage_features=leakage_features
    )
    
    return model, y_pred, rmse, mae, r2, feature_names

def prepare_forecast_data(df, year, feature_cols, encoders_dict=None):
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
    encoders_dict : dict, optional
        Dictionary of encoders for categorical features
        
    Returns:
    --------
    forecast_df : numpy.ndarray
        Dense array prepared for forecasting
    """
    # Filter data for the current year
    if isinstance(df, pd.DataFrame):
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
        # Ensure all columns in feature_cols exist in the dataframe
        existing_cols = [col for col in feature_cols if col in current_year_data.columns]
        missing_cols = set(feature_cols) - set(existing_cols)
        
        if missing_cols:
            print(f"Warning: Missing columns in forecast data: {missing_cols}")
            # Add missing columns with zeros
            for col in missing_cols:
                current_year_data[col] = 0.0
        
        current_year_data_subset = current_year_data[feature_cols].copy()
        
        # Process with our data preparation functions and convert to dense if needed
        if encoders_dict:
            X, _ = prepare_sparse_input(current_year_data_subset, feature_cols, encoders=encoders_dict)
            return X.toarray()
        else:
            # Ensure all columns are float64
            for col in current_year_data_subset.columns:
                current_year_data_subset[col] = pd.to_numeric(current_year_data_subset[col], errors='coerce').fillna(0).astype(np.float64)
            return current_year_data_subset.values
    else:
        # Input is already a dense array or similar
        # Just ensure it's a numpy array with float64 dtype
        return np.asarray(df, dtype=np.float64)

# Choose the last pre-COVID year as test year (2019 if available)
target_test_year = 2019  # Last pre-COVID year
if target_test_year in available_years:
    test_year = target_test_year
else:
    # Find the highest year before 2020 (COVID start)
    pre_covid_years = [year for year in available_years if year < 2020]
    if pre_covid_years:
        test_year = max(pre_covid_years)
    else:
        # If no pre-COVID years, use the earliest COVID year
        test_year = min([year for year in available_years if year >= 2020])

print(f"Using {test_year} as test year to avoid COVID anomalies in evaluation")

train_df = df[df['Year'] < test_year].copy()
test_df = df[df['Year'] == test_year].copy()

# Take a smaller sample for testing
sample_size = 100000  # Reduced from 500,000 to manage memory better
train_df = train_df.sample(n=min(len(train_df), sample_size), random_state=42)
test_df = test_df.sample(n=min(len(test_df), sample_size), random_state=42)
print(f"Using reduced dataset: Train={len(train_df)}, Test={len(test_df)}")

print(f"Training data years: {train_df['Year'].min()} to {train_df['Year'].max()}")
print(f"Testing data year: {test_df['Year'].unique()}")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Prepare common features for tasks 1a, 1b, and 1c
exclude_cols = ['Discharges', 'Mean Cost', 'Median Cost', 'Mean Charge', 'Median Charge', 'Facility Name', 'Facility_DRG_Key', 'Year_Cat']
include_cols = [col for col in df.columns if col not in exclude_cols]

# Explicitly add the new temporal and health event features
important_features = [
    'Prev_Year_Discharges', 
    'Prev2_Year_Discharges',
    'YoY_Discharges_Change',
    'Discharge_Trend',
    'Discharge_Trend_Direction',
    'Consistent_Growth',
    'COVID_Period',
    'COVID_Respiratory_Impact',
    'Prev_Year_Impact',
    'Facility_Avg_Discharges',
    'DRG_Avg_Discharges',
    'Severity_Avg_Discharges'
]

# Make sure these features exist in the dataframe
include_cols = [col for col in include_cols if col in df.columns]
important_features = [col for col in important_features if col in df.columns]

# Ensure important features are included
for feature in important_features:
    if feature not in include_cols and feature in df.columns:
        include_cols.append(feature)

# Run Task 1a: Predict Discharges
model_1a, y_pred_1a, rmse_1a, mae_1a, r2_1a, feature_names_1a = run_task1a_discharge_prediction(
    train_df, test_df, include_cols, encoders
)

# Run Task 1b: Predict Median Costs
model_1b, y_pred_1b, rmse_1b, mae_1b, r2_1b, feature_names_1b = run_task1b_median_cost_prediction(
    train_df, test_df, include_cols, encoders
)

# Run Task 1c: Predict Median Charges
model_1c, y_pred_1c, rmse_1c, mae_1c, r2_1c, feature_names_1c = run_task1c_median_charge_prediction(
    train_df, test_df, include_cols, encoders
)

# Run Task 2: Predict Total DRG Discharges
model_2, y_pred_2, rmse_2, mae_2, r2_2, drg_test = run_task2_total_drg_prediction(
    df_orig, test_year
)

# Run Task 3: Predict Mean Cost
model_3, y_pred_3, rmse_3, mae_3, r2_3, feature_names_3 = run_task3_mean_cost_prediction(
    train_df, test_df, include_cols, encoders
)

print("\nStep 4: Model Evaluation & Interpretation")
# Skip SHAP analysis for simplicity

# Create a comprehensive results table
print("\nModel Performance Summary:")
print("="*80)
results_df = pd.DataFrame({
    'Task': [
        'Task 1a: Discharges Prediction', 
        'Task 1b: Median Costs Prediction', 
        'Task 1c: Median Charges Prediction', 
        'Task 2: Total DRG Discharges', 
        'Task 3: Mean Cost Prediction'
    ],
    'RMSE': [rmse_1a, rmse_1b, rmse_1c, rmse_2, rmse_3],
    'MAE': [mae_1a, mae_1b, mae_1c, mae_2, mae_3],
    'R²': [r2_1a, r2_1b, r2_1c, r2_2, r2_3]
})

# Format for better display
results_df['RMSE'] = results_df['RMSE'].map(lambda x: f"{x:.2f}")
results_df['MAE'] = results_df['MAE'].map(lambda x: f"{x:.2f}")
results_df['R²'] = results_df['R²'].map(lambda x: f"{x:.4f}")

print(results_df.to_string(index=False))

# Save model performance summary to CSV
results_df.to_csv('model_performance_summary.csv', index=False)
print("\nModel performance summary saved to 'model_performance_summary.csv'")

# Time tracking
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

print("\nAnalysis completed successfully!") 