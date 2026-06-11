# PSURidgeRegressionModel

This project implements Ridge Regression with cross-validation and learning curve analysis for predictive modeling. It provides tools for hyperparameter tuning, model evaluation, and visualization of regression performance metrics.

## Capabilities

- **Ridge Regression**: Implements L2-regularized linear regression for handling multicollinearity and overfitting
- **10-Fold Cross-Validation**: Evaluates model performance across different regularization strengths (alpha values from 1e-4 to 1e4)
- **Feature Standardization**: Compares model performance with and without standardized covariates
- **Learning Curves**: Visualizes training and validation MSE across different training set sizes
- **Optimal Alpha Selection**: Automatically identifies the best regularization parameter for model optimization
- **Test Set Evaluation**: Assesses model generalization on held-out test data

## Usage

### Terminal MAC Run Script

```bash
# Run cross-validation for Ridge regression (basic)
python3 Ridge_CV.py

# Run cross-validation with standardization comparison
python3 Ridge_CV_Standardized.py

# Generate learning curves with optimal alpha
python3 Learning_Curves.py

# Evaluate model on test dataset
python3 Test_Evaluation.py
```

## Use Cases

### Hyperparameter Tuning
Systematically identify optimal regularization parameters through cross-validation, comparing unscaled and standardized feature performance.

### Model Diagnostics
Visualize learning curves to assess model bias-variance tradeoff, determine if more data is needed, and evaluate convergence behavior.

### Predictive Modeling
Train Ridge regression models on datasets with multiple features (29 covariates) to predict continuous target variables.

### Regression Analysis
Compare MSE metrics across training, validation, and test sets to understand model generalization.

## Research Purposes

Designed for research purposes in regression analysis and regularization techniques. Penn State University (PSU), IST 557 Data Mining. Fall 2025.