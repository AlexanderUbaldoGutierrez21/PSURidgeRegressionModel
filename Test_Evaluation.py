import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# LOAD TRAINING DATA
train_data = pd.read_csv('DataTrain_HW3Problem1.csv')

# SEPARATE FEATURES AND TARGET
X_train = train_data.drop('y', axis=1)
y_train = train_data['y']

# STANDARDIZE TRAINING FEATURES
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_scaled = scaler.fit_transform(X_train)

# OPTIMAL ALPHA FROM SCALED CV
optimal_alpha = 11.513954

# TRAIN RIDGE MODEL ON FULL TRAINING DATA
model = Ridge(alpha=optimal_alpha)
model.fit(X_train_scaled, y_train)

# LOAD TEST DATA
test_data = pd.read_csv('DataTest_HW3Problem1.csv')

# SEPARATE FEATURES AND TARGET
X_test = test_data.drop('y', axis=1)
y_test = test_data['y']

# STANDARDIZE TEST FEATURES USING TRAINING SCALER
X_test_scaled = scaler.transform(X_test)

# PREDICT ON TEST SET
y_test_pred = model.predict(X_test_scaled)

# CALCULATE MSE
test_mse = mean_squared_error(y_test, y_test_pred)

# PRINT RESULTS
print(f'Optimal alpha used: {optimal_alpha}')
print(f'Test MSE: {test_mse:.4f}')

print("\nModel Evaluation:")
print("- Compare to validation MSE from CV (16.07) and learning curves (14.48).")
print("- If test MSE is similar, model generalizes well; if higher, may indicate overfitting or data issues.")
print("- Happiness depends on context: for this dataset, MSE around 15-20 may be acceptable if baseline is higher.")