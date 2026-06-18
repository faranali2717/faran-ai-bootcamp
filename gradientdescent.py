import numpy as np
import matplotlib.pyplot as plt 
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


diabetes= datasets.load_diabetes()
x=diabetes.data
y=diabetes.target

print(f" total samples: {x.shape[0]}")
print (f"total features: {x.shape[1]}")

print("\nTrain-Test-Split(Data)")

X_train, X_test , y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)


print("\nFeature scaling")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_with_intercept = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_with_intercept = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]




def gradient_descent(X, y, learning_rate=0.1, num_epochs=1000):
    """
    Batch Gradient Descent for Linear Regression
    
    Parameters:
    X: Training data (with intercept)
    y: Target values
    learning_rate: Step size (alpha)
    num_epochs: Number of iterations
    
    Returns:
    theta: Learned parameters
    cost_history: List of cost values per epoch
    """
    m = X.shape[0]  # Number of training examples
    n = X.shape[1]  # Number of features (including intercept)
    
    # Initialize theta to zeros
    theta = np.zeros(n)
    
    # Lists to store history
    cost_history = []
    epoch_history = []
    
    for epoch in range(num_epochs):
        # 1. Make predictions
        predictions = X @ theta
        
        # 2. Calculate error
        error = predictions - y
        
        # 3. Calculate cost (MSE with 1/2)
        cost = (1 / (2 * m)) * np.sum(error ** 2)
        cost_history.append(cost)
        epoch_history.append(epoch)
        
        # 4. Calculate gradient
        gradient = (1 / m) * (X.T @ error)
        
        # 5. Update parameters (SIMULTANEOUSLY!)
        theta = theta - learning_rate * gradient
        
        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}: Cost = {cost:.6f}")
    
    return theta, cost_history, epoch_history


print("\n[STEP 6] Training model...")

# Try different learning rates
learning_rates = [0.01, 0.05, 0.1, 0.3, 0.5]
results = {}

print("\nTrying different learning rates...")


for lr in learning_rates:
    print(f"\n--- Learning Rate: {lr} ---")
    theta, cost_history, epoch_history = gradient_descent(
        X_train_with_intercept, y_train, 
        learning_rate=lr, num_epochs=1000
    )
    results[lr] = {
        'theta': theta,
        'cost_history': cost_history,
        'epoch_history': epoch_history
    }


   

# Find best learning rate (lowest final cost)
best_lr = None
best_final_cost = float('inf')

for lr, data in results.items():
    final_cost = data['cost_history'][-1]
    if final_cost < best_final_cost:
        best_final_cost = final_cost
        best_lr = lr

print(f"\nBest learning rate: {best_lr}")
print(f"Final cost: {best_final_cost:.6f}")

# Get best model's parameters
best_theta = results[best_lr]['theta']
best_cost_history = results[best_lr]['cost_history']
best_epoch_history = results[best_lr]['epoch_history']

print(f"\nBest theta values:")
for i in range(len(best_theta)):
    if i == 0:# STEP 9: CALCULATE METRICS
        print(f"  θ₀ (intercept) = {best_theta[i]:.4f}")
    else:
        print(f"  θ_{i} = {best_theta[i]:.4f}")


#  CALCULATE METRICS

y_train_pred = X_train_with_intercept @ best_theta
y_test_pred = X_test_with_intercept @ best_theta

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)


# Plot 1: Loss vs Epochs for different learning rates
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for lr, data in results.items():
    plt.plot(data['epoch_history'], data['cost_history'], 
             label=f'α = {lr}', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Cost J(θ)')
plt.title('Loss vs Epochs for Different Learning Rates')
plt.legend()
plt.grid(True, alpha=0.3)

# Zoom in on first 100 epochs to see convergence
plt.subplot(1, 2, 2)
for lr, data in results.items():
    plt.plot(data['epoch_history'][:100], data['cost_history'][:100], 
             label=f'α = {lr}', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Cost J(θ)')
plt.title('Loss vs Epochs (First 100 Epochs)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Plot 2: Best model's convergence
plt.figure(figsize=(10, 6))
plt.plot(best_epoch_history, best_cost_history, 'b-', linewidth=2, label=f'α = {best_lr}')
plt.axhline(y=best_cost_history[-1], color='r', linestyle='--', 
            label=f'Final Cost = {best_cost_history[-1]:.2f}')
plt.xlabel('Epochs')
plt.ylabel('Cost J(θ)')
plt.title(f'Gradient Descent Convergence (Learning Rate = {best_lr})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# PREDICTIONS VS ACTUAL
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.6, color='blue', edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Test Set: Predictions vs Actual\nR² = {test_r2:.4f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
residuals = y_test - y_test_pred
plt.hist(residuals, bins=30, color='green', edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Frequency')
plt.title(f'Error Distribution\nMean Error = {residuals.mean():.2f}')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Normal equation for comparison
theta_ne = np.linalg.inv(X_train_with_intercept.T @ X_train_with_intercept) @ X_train_with_intercept.T @ y_train

print("\nParameter Comparison:")
print("Feature\t\tGD (Best)\tNE\t\tDifference")
print("-"*60)
for i in range(len(best_theta)):
    if i == 0:
        name = "Intercept"
    else:
        name = diabetes.feature_names[i-1]
    diff = abs(best_theta[i] - theta_ne[i])
    print(f"{name:12}\t{best_theta[i]:8.4f}\t{theta_ne[i]:8.4f}\t{diff:8.4f}")
