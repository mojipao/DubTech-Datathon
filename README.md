# SPARCS Healthcare Data Analysis and Prediction

## Abstract
This project implements machine learning models to predict key healthcare metrics from the New York State's SPARCS (Statewide Planning and Research Cooperative System) dataset. By analyzing hospital discharge data, we developed predictive models for discharges, costs, and charges at both the individual hospital/DRG level and statewide aggregates. Our approach employs advanced feature engineering, careful prevention of data leakage, and interpretable machine learning techniques with SHAP analysis. The models achieve strong predictive performance with R² values ranging from 0.85 to 0.97 across different prediction tasks, providing valuable insights for healthcare planning, resource allocation, and cost management.

## Introduction
The SPARCS dataset contains valuable information about hospital discharges, associated costs, and charges across New York State. By applying machine learning to this dataset, we aim to:

1. Predict hospital-specific and DRG-specific metrics for the upcoming year
2. Forecast total DRG discharges across New York State
3. Analyze the factors that influence the cost of hospital stays

These predictions can help healthcare administrators, policy makers, and insurance companies make informed decisions about resource allocation, budgeting, and healthcare planning.

## Data and Methods

### Dataset
The analysis uses the Hospital Inpatient Discharges (SPARCS De-Identified) dataset from the New York State Department of Health, which includes:
- Hospital facility information
- Diagnosis-related group (DRG) codes
- Severity of illness
- Medical/surgical classification
- Discharge counts
- Cost and charge metrics

### Preprocessing
The preprocessing pipeline includes:
- Converting data types (categorical and numerical)
- Handling missing values with appropriate imputation
- Cleaning and validating logical relationships (e.g., costs vs. charges)
- Detecting and handling anomalies and outliers

### Feature Engineering
We implemented extensive feature engineering to capture:
1. **Temporal patterns**: Year-over-year changes, trends, and growth indicators
2. **Facility-level aggregates**: Average costs, charges, and discharges by facility
3. **DRG-level aggregates**: Average metrics by diagnosis group
4. **Severity-based indicators**: Patterns based on illness severity
5. **Interaction effects**: Combined impact of multiple factors

### Modeling Approach
- **Algorithm**: Gradient Boosting (HistGradientBoostingRegressor)
- **Validation**: Temporal validation using earlier years for training and a later year for testing
- **Feature Leakage Prevention**: Careful exclusion of related metrics in each prediction task
- **Interpretability**: SHAP analysis for understanding model decisions

### Implementation
The analysis is implemented in Python, using:
- scikit-learn for modeling and evaluation
- pandas for data manipulation
- matplotlib and seaborn for visualization
- SHAP for model interpretation

## Results and Discussion

### Task 1: Hospital & DRG Specific Metrics Prediction

#### Discharges Prediction
- **Performance**: R² = 0.85, RMSE = 30.61, MAE = 2.84
- **Key Findings**: 
  - Historical discharge patterns are strong predictors of future discharges
  - Facility size and DRG type significantly influence discharge counts
  - Year-over-year trends provide valuable predictive signals

#### Median Costs Prediction
- **Performance**: R² = 0.97, RMSE = 3699.90, MAE = 1400.61
- **Key Findings**:
  - Severity of illness is a dominant factor in determining costs
  - Facility-specific factors strongly influence costs, suggesting different cost structures across hospitals
  - DRG code is highly predictive, reflecting the standardized cost expectations for specific procedures

#### Median Charges Prediction
- **Performance**: R² = 0.94, RMSE = 16130.06, MAE = 7419.09
- **Key Findings**:
  - Charge patterns show greater variability than costs (higher RMSE)
  - Facility factors have even stronger influence than with costs, suggesting different charging strategies
  - Temporal trends reveal evolving charging patterns over time

### Task 2: Total DRG Discharges Prediction for New York State
- **Performance**: R² = 0.91, RMSE = 4661.71, MAE = 905.24
- **Key Findings**:
  - Previous year's discharge counts are strong predictors of future totals
  - Year-over-year change patterns provide significant predictive value
  - Some DRG codes show more stable patterns than others, suggesting different levels of predictability

### Task 3: Mean Cost Prediction Using Hospital and Clinical Features
- **Performance**: R² = 0.97, RMSE = 3983.35, MAE = 1505.70
- **Key Findings**:
  - DRG code and severity of illness are the strongest predictors
  - Medical vs. surgical classification significantly impacts costs
  - Hospital-specific factors indicate substantial variation in cost structure across facilities
  - Year trends show general cost increases over time, with some exceptions

### SHAP Analysis Insights
The SHAP analysis reveals how different features contribute to predictions:

1. **For Discharge Prediction**:
   - Previous year's discharges have the highest impact
   - Hospital size/capacity indicators strongly influence predictions
   - DRG-specific historical patterns show different growth trajectories

2. **For Cost Predictions**:
   - Severity of illness shows a clear positive correlation with costs
   - Hospital factors reveal institutional cost differences
   - DRG codes demonstrate procedure-specific cost patterns
   - Year features capture inflation and policy changes over time

3. **For Charge Predictions**:
   - Hospital factors have even stronger influence than on costs
   - Regional patterns emerge, showing geographic variations in charging
   - Correlation between costs and charges varies across hospitals

## Conclusions and Implications

Our predictive models demonstrate strong performance across all tasks, with particularly high accuracy in cost and charge predictions. These results have several important implications:

1. **For Hospital Administrators**:
   - Reliable forecasting of patient volumes helps with staffing and resource planning
   - Understanding cost drivers enables more effective budgeting
   - Facility-specific factors identified can guide operational improvements

2. **For Healthcare Policy**:
   - Statewide discharge predictions support system-level planning
   - Cost variation insights can inform standardization efforts
   - Identified temporal trends help evaluate policy impact

3. **For Insurance and Payment Systems**:
   - Accurate cost predictions aid in setting appropriate reimbursement rates
   - Understanding charge-to-cost ratios across facilities reveals billing patterns
   - DRG-specific insights support case-mix adjustments

## Future Work
1. Incorporate additional data sources (demographics, social determinants of health)
2. Extend predictions to longer time horizons
3. Develop interactive tools for scenario planning
4. Implement more granular predictions at the patient level, where data permits
5. Extend analysis to other healthcare metrics beyond the current focus

## Requirements and Usage

### Installation and Dependencies

1. **Ensure Python 3.7+ is installed**
   ```bash
   python --version
   ```



2. **Install dependencies using pip**
   ```bash
   # Install all required packages
   pip install -r requirements.txt
   
   # For virtual environment users (recommended)
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Note for Mac users**: 
   LightGBM is optional and disabled by default on macOS due to `libomp` dependency issues. The script will automatically fall back to using HistGradientBoostingRegressor if LightGBM is not available.

4. **Download the dataset**
   The analysis requires the SPARCS dataset file. Download "Hospital_Inpatient_Discharges__SPARCS_De-Identified___Cost_Transparency__Beginning_2009.csv" and place it in the project directory.

### Running the Analysis

Once dependencies are installed, run the full analysis with:
```bash
python sparcs_analysis.py
```

The script will automatically:
- Load and preprocess the SPARCS dataset
- Engineer features for prediction
- Train models for all prediction tasks
- Evaluate model performance
- Generate SHAP visualizations for model interpretability
- Create summary outputs and visualizations

### Output Files
The script will generate several output files for analysis:

#### Performance Metrics
- `model_performance_summary.csv`: Summary of performance metrics (RMSE, MAE, R²) for all models

#### Visualizations
- `shap_summary_*.png`: SHAP summary plots showing feature importance for each model
- `shap_dependence_*.png`: SHAP dependence plots showing relationships between features and predictions
- `predictions_vs_actual.png`: Scatter plots comparing predicted vs actual values for each model
- `feature_importance_*.png`: Feature importance plots for models with built-in importance attributes

#### Forecast Files
- `forecast_hospital_drg_metrics.csv`: Predicted hospital and DRG-specific metrics for the next year
  - Includes predicted discharges, median costs, and median charges
- `forecast_total_drg_discharges.csv`: Predicted total discharges by DRG type for New York State
- `forecast_mean_costs.csv`: Predicted mean costs per discharge for the next year

These files can be used for further analysis, planning, and integration with healthcare dashboards and decision support systems. 