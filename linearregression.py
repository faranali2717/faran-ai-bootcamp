import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("Linear Regression using Normal Equation")
print("-" * 40)

# Load the diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# Add a bias column (column of 1s) for the intercept
X = np.c_[np.ones((X.shape[0], 1)), X]

# Split into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# Solve for theta using the Normal Equation: θ = (XᵀX)⁻¹Xᵀy
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

print("\nLearned coefficients:")
feature_names = ['intercept'] + list(diabetes.feature_names)
for name, val in zip(feature_names, theta):
    print(f"  {name:>12} = {val:.4f}")

# Predict
y_train_pred = X_train @ theta
y_test_pred  = X_test  @ theta

# Metrics
train_r2  = r2_score(y_train, y_train_pred)
test_r2   = r2_score(y_test,  y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse  = mean_squared_error(y_test,  y_test_pred)
residuals = y_test - y_test_pred

print(f"\n{'':15} {'Train':>10} {'Test':>10}")
print(f"{'MSE':15} {train_mse:>10.2f} {test_mse:>10.2f}")
print(f"{'RMSE':15} {np.sqrt(train_mse):>10.2f} {np.sqrt(test_mse):>10.2f}")
print(f"{'R²':15} {train_r2:>10.4f} {test_r2:>10.4f}")

# First 10 predictions vs actuals
print("\nSample predictions (first 10 from test set):")
print(f"  {'Predicted':>10}  {'Actual':>8}  {'Error':>8}")
for pred, act in zip(y_test_pred[:10], y_test[:10]):
    print(f"  {pred:>10.2f}  {act:>8.2f}  {pred - act:>+8.2f}")

# Plot 1: Predictions vs Actual + Residuals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(y_test, y_test_pred, alpha=0.6, color='steelblue', edgecolors='black')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set(xlabel='Actual', ylabel='Predicted', title=f'Predictions vs Actual  (R² = {test_r2:.4f})')
ax1.grid(True, alpha=0.3)

ax2.hist(residuals, bins=30, color='seagreen', edgecolor='black', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', lw=2)
ax2.set(xlabel='Residual (Actual − Predicted)', ylabel='Count',
        title=f'Residual Distribution  (mean = {residuals.mean():.2f})')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predictions.png')
plt.close()

# Plot 2: Feature importance by absolute coefficient value
theta_abs = np.abs(theta)
order = np.argsort(theta_abs)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(len(theta)), theta_abs[order], color='mediumpurple', alpha=0.8)
plt.xticks(range(len(theta)), [feature_names[i] for i in order], rotation=45, ha='right')
plt.ylabel('|Coefficient|')
plt.title('Feature Importance')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print(f"""
Summary
-------
The model explains {test_r2*100:.1f}% of the variance in the test set (R² = {test_r2:.4f}).
Average prediction error: {abs(residuals.mean()):.2f} units  |  RMSE: {np.sqrt(test_mse):.2f}
Plots saved → predictions.png, feature_importance.png
""")
