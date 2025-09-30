import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

# UN-SCALED LEARNING CURVES
train_mses_un = []
val_mses_un = []
for alpha in alphas:
    tm, vm = [], []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        tm.append(mean_squared_error(y_train, y_train_pred))
        vm.append(mean_squared_error(y_val, y_val_pred))

    train_mses_un.append(np.mean(tm))
    val_mses_un.append(np.mean(vm))

# STANDARDIZED LEARNING CURVES (PER-FOLD SCALING VIA PIPELINE)
train_mses_sc = []
val_mses_sc = []
for alpha in alphas:
    tm, vm = [], []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('ridge', Ridge(alpha=alpha))
        ])
        pipe.fit(X_train, y_train)

        y_train_pred = pipe.predict(X_train)
        y_val_pred = pipe.predict(X_val)

        tm.append(mean_squared_error(y_train, y_train_pred))
        vm.append(mean_squared_error(y_val, y_val_pred))

    train_mses_sc.append(np.mean(tm))
    val_mses_sc.append(np.mean(vm))

# FIND OPTIMAL ALPHAS
opt_un_idx = int(np.argmin(val_mses_un))
opt_sc_idx = int(np.argmin(val_mses_sc))
opt_alpha_un = alphas[opt_un_idx]
opt_alpha_sc = alphas[opt_sc_idx]
opt_val_un = float(val_mses_un[opt_un_idx])
opt_val_sc = float(val_mses_sc[opt_sc_idx])

# OVERLAY COMPARISON PLOT
plt.figure(figsize=(10, 6))
plt.plot(alphas, train_mses_un, label='Unscaled - Train MSE', color='C0', linestyle='-')
plt.plot(alphas, val_mses_un, label='Unscaled - Val MSE', color='C1', linestyle='-')
plt.plot(alphas, train_mses_sc, label='Scaled - Train MSE', color='C0', linestyle='--')
plt.plot(alphas, val_mses_sc, label='Scaled - Val MSE', color='C1', linestyle='--')
plt.xscale('log')
plt.xlabel('Alpha (Log Scale)')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Regression Learning Curves: Unscaled vs Standardized (10-fold CV)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Ridge_Learning_Curves_Compare.png', dpi=300, bbox_inches='tight')
plt.show()

# SCALED-ONLY PLOT (PART B)
plt.figure(figsize=(10, 6))
plt.plot(alphas, train_mses_sc, label='Mean Training MSE (Scaled)', marker='o')
plt.plot(alphas, val_mses_sc, label='Mean Validation MSE (Scaled)', marker='s')
plt.xscale('log')
plt.xlabel('Alpha (Log Scale)')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curves for Ridge Regression with Standardized Covariates (10-fold CV)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Ridge_Learning_Curves_Standardized.png', dpi=300, bbox_inches='tight')
plt.show()

# PRINT COMPARISON SUMMARY
print(f'Unscaled optimal alpha: {opt_alpha_un:.6f} (val MSE={opt_val_un:.6f})')
print(f'Scaled optimal alpha:   {opt_alpha_sc:.6f} (val MSE={opt_val_sc:.6f})')
delta = opt_val_un - opt_val_sc
print(f'Delta (unscaled - scaled) optimal val MSE: {delta:.6f}')
print("\nInterpretation:")
print("- Standardizing covariates enforces comparable feature scales so Ridge penalizes coefficients uniformly.")
print("- After scaling, the optimal alpha often shifts (commonly larger) and validation error can improve when the original features have heterogeneous scales.")
print("- Inspect the saved plots to compare curve shapes and the locations of the minima between unscaled and scaled settings.")