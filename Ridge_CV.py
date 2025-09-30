import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# LOAD TRAINING DATA
train_data = pd.read_csv('DataTrain_HW3Problem1.csv')

# SEPARATE FEATURES AND TARGET
X = train_data.drop('y', axis=1)
y = train_data['y']

# DEFINE RANGE OF ALPHAS (LOG SPACED FROM 1e-4 TO 1e4)
alphas = np.logspace(-4, 4, 50)

# 10-FOLD CROSS-VALIDATION
kf = KFold(n_splits=10, shuffle=True, random_state=42)

train_mses = []
val_mses = []

for alpha in alphas:
    train_mse_folds = []
    val_mse_folds = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # FIT RIDGE MODEL 
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        # PREDICT ON TRAINING FOLD
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mse_folds.append(train_mse)

        # PREDICT ON VALIDATION FOLD
        y_val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_mse_folds.append(val_mse)

    # AVERAGE OVER FOLDS
    train_mses.append(np.mean(train_mse_folds))
    val_mses.append(np.mean(val_mse_folds))

# FIND OPTIMAL ALPHA (MINIMUN VALIDATION MSE)
optimal_idx = np.argmin(val_mses)
optimal_alpha = alphas[optimal_idx]

# PLOT LEARNING CURVES
plt.figure(figsize=(10, 6))
plt.plot(alphas, train_mses, label='Mean Training MSE', marker='o')
plt.plot(alphas, val_mses, label='Mean Validation MSE', marker='s')
plt.xscale('log')
plt.xlabel('Alpha (Log Scale)')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curves for Ridge Regression (10-fold CV)')
plt.legend()
plt.grid(True)
plt.savefig('Ridge_Learning_Curves.png', dpi=300, bbox_inches='tight')
plt.show()

# PRINT OPTIMAL ALPHA
print(f'Optimal alpha: {optimal_alpha:.6f}')

# DESCRIPTION
print("\nDescription:")
print("The plot shows the mean training MSE and mean validation MSE as a function of the regularization parameter alpha.")
print("Alpha is plotted on a logarithmic scale, with low penalization (small alpha) on the left and high penalization (large alpha) on the right.")
print("As alpha increases, the model becomes more regularized, which typically increases the training error but can decrease overfitting.")
print("The validation error initially decreases as alpha increases from very small values, reaches a minimum, and then increases.")
print("The optimal alpha is chosen as the value that minimizes the mean validation MSE, which is {:.6f}.".format(optimal_alpha))
print("This alpha provides the best balance between bias and variance for this dataset.")