import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# LOAD TRAINING DATA
data = pd.read_csv('DataTrain_HW3Problem1.csv')

# SEPARATE FEATURES AND TARGET
X = data.drop('y', axis=1)
y = data['y']

# SPLIT INTO TRAIN AND VALIDATION SETS (80/20)
X_train_full, X_val, y_train_full, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# STANDARDIZE FEATURES (USING TRAINING SET STATS)
scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_val_scaled = scaler.transform(X_val)

# OPTIMAL ALPHA FROM PREVIOUS SCALED CV (11.513954)
optimal_alpha = 11.513954

# LEARNING CURVE: TRAIN ON INCREASING SUBSETS
train_sizes = np.linspace(0.1, 1.0, 10)  # 10% to 100% of training data
train_mses = []
val_mses = []

for size in train_sizes:
    n_samples = int(size * len(X_train_full_scaled))
    X_subset = X_train_full_scaled[:n_samples]
    y_subset = y_train_full[:n_samples]

    # TRAIN RIDGE MODEL
    model = Ridge(alpha=optimal_alpha)
    model.fit(X_subset, y_subset)

    # PREDICT ON TRAINING SUBSET
    y_train_pred = model.predict(X_subset)
    train_mse = mean_squared_error(y_subset, y_train_pred)
    train_mses.append(train_mse)

    # PREDICT ON VALIDATION SET
    y_val_pred = model.predict(X_val_scaled)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mses.append(val_mse)

# PLOT LEARNING CURVES
plt.figure(figsize=(10, 6))
plt.plot(train_sizes * 100, train_mses, label='Training MSE', marker='o')
plt.plot(train_sizes * 100, val_mses, label='Validation MSE', marker='s')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curves for Ridge Regression (Standardized, alpha=11.51)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Learning_Curves.png', dpi=300, bbox_inches='tight')
plt.show()

# PRINT SUMMARY
print(f'Optimal alpha used: {optimal_alpha}')
print(f'Final training MSE: {train_mses[-1]:.4f}')
print(f'Final validation MSE: {val_mses[-1]:.4f}')
print(f'Gap at 100%: {val_mses[-1] - train_mses[-1]:.4f}')

print("\nInterpretation:")
print("- If curves converge (gap stabilizes), data is sufficient.")
print("- High gap indicates underfitting or need for more data.")
print("- Inspect the plot for convergence behavior.")