# Crab Age Prediction Model

This project focuses on predicting the age of crabs based on physical measurements and characteristics. We use a training dataset to build a predictive model, perform data exploration, clean the data, analyze it, and test the model with a separate dataset. Various statistical techniques and diagnostics are applied to ensure model reliability.

## Table of Contents
1. [Requirements](#requirements)
2. [Dataset Loading](#dataset-loading)
3. [Data Exploration and Cleaning](#data-exploration-and-cleaning)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Encoding and Multicollinearity Check](#encoding-and-multicollinearity-check)
6. [Multiple Linear Regression Model](#multiple-linear-regression-model)
7. [Model Diagnostics](#model-diagnostics)
8. [Removing Outliers](#removing-outliers)
9. [Testing and Predictions](#testing-and-predictions)

## Requirements

The following Python libraries are required:

```python
numpy
pandas
matplotlib
seaborn
statsmodels
scikit-learn
```

Install the libraries via pip if necessary:
```bash
pip install numpy pandas matplotlib seaborn statsmodels scikit-learn
```

## Dataset Loading

Load the training dataset:
```python
df = pd.read_csv("/path/to/training_dataset.csv")
```

## Data Exploration and Cleaning

1. **Check for Duplicates and Missing Values**
   ```python
   duplicates = df.duplicated().sum()
   missing_values = df.isnull().sum()
   ```
2. **Data Types and Separation**
   - Separate numerical and categorical data for further analysis.

3. **Drop Unnecessary Columns**
   ```python
   df = df.drop(columns=['id'])
   ```

## Exploratory Data Analysis

1. **Pie Chart of Categorical Variable (Sex)**
2. **Histograms with Normal Distribution Approximation**
   - Visualize the distribution of numerical features against a normal approximation.

## Encoding and Multicollinearity Check

- **One-Hot Encoding**: Convert categorical variables into numerical format using `OneHotEncoder`.
- **Correlation Matrix**: Visualize the correlation among features to detect multicollinearity.
- **Variance Inflation Factor (VIF)**: Quantify multicollinearity among predictors.

## Multiple Linear Regression Model

1. **Variable Selection**: Use selected variables for the predictive model.
2. **Fit Model**: Use Ordinary Least Squares (OLS) regression to fit the model.
   ```python
   model = sm.OLS(y, X).fit()
   print(model.summary())
   ```

## Model Diagnostics

1. **Residuals vs. Fitted Values**: Plot residuals to check for homoscedasticity.
2. **Q-Q Plot**: Assess normality of residuals.

## Removing Outliers

1. Identify outliers with high leverage and filter them out to improve model robustness.

## Testing and Predictions

1. **Load Test Dataset**
   ```python
   test_df = pd.read_csv('/path/to/test_dataset.csv')
   ```
2. **Encode and Prepare Test Data**
3. **Predict Age**: Use the fitted model to predict crab ages in the test dataset.
   ```python
   predicted_log_age = model_filtered.predict(features)
   predicted_age = np.exp(predicted_log_age)
   ```
4. **Display Predicted Ages**
