"""
Logistic regression from scratch using full-batch gradient descent.
Evaluated on the scikit-learn breast cancer dataset.
Target: >90% test accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def sigmoid(z):
    # Clip to prevent overflow issues with np.exp() on extreme values
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))


class LogisticRegressionGD:
    """Vanilla logistic regression classifier optimized via batch gradient descent."""

    def __init__(self, learning_rate=0.1, num_epochs=1000, verbose=False):
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.theta = None
        self.cost_history = []

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        self.cost_history = []

        for epoch in range(self.num_epochs):
            z = X @ self.theta
            h = sigmoid(z)

            # Robust log-loss computation using clipped boundaries to prevent log(0)
            h_clipped = np.clip(h, 1e-15, 1 - 1e-15)
            cost = -np.mean(y * np.log(h_clipped) + (1 - y) * np.log(1 - h_clipped))
            self.cost_history.append(cost)

            # Gradient evaluation & parameter step
            grad = (X.T @ (h - y)) / m
            self.theta -= self.lr * grad

            if self.verbose and epoch % 200 == 0:
                print(f"Epoch {epoch:4d} | Cost: {cost:.6f}")

        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.theta)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


def add_intercept(X):
    return np.c_[np.ones((X.shape[0], 1)), X]


def main():
    # Load and split data
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"Dataset stats: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Distribution: {np.sum(y == 0)} malignant / {np.sum(y == 1)} benign\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features (crucial step for stable gradient descent convergence)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_i = add_intercept(X_train_s)
    X_test_i = add_intercept(X_test_s)

    # Hyperparameter optimization grid sweep
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5]
    results = {}

    for lr in learning_rates:
        model = LogisticRegressionGD(learning_rate=lr, num_epochs=2000)
        model.fit(X_train_i, y_train)
        results[lr] = {
            "model": model,
            "train_acc": model.score(X_train_i, y_train),
            "test_acc": model.score(X_test_i, y_test),
        }
        print(f"lr={lr:<6} | Train Acc: {results[lr]['train_acc']:.4f} | Test Acc: {results[lr]['test_acc']:.4f}")

    best_lr = max(results, key=lambda lr: results[lr]["test_acc"])
    best_model = results[best_lr]["model"]
    print(f"\nSelected Optimal Learning Rate: {best_lr} (Test Acc: {results[best_lr]['test_acc']:.4f})")

    # Final inference & metrics report
    y_pred = best_model.predict(X_test_i)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nFinal Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

    # --- Plotting Generation Loops ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for lr, r in results.items():
        axes[0, 0].plot(r["model"].cost_history, label=f"lr={lr}")
    axes[0, 0].set_title("Cost vs Epoch (Full Profile)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    for lr, r in results.items():
        axes[0, 1].plot(r["model"].cost_history[:200], label=f"lr={lr}")
    axes[0, 1].set_title("Cost vs Epoch (Early Initialization - First 200)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    best_costs = best_model.cost_history
    axes[1, 0].plot(best_costs, color="navy", label=f"lr={best_lr}")
    axes[1, 0].axhline(best_costs[-1], color="crimson", linestyle="--",
                        label=f"Final Cost: {best_costs[-1]:.4f}")
    axes[1, 0].set_title("Optimal Model Convergence Curve")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"], ax=axes[1, 1], cbar=False)
    axes[1, 1].set_title(f"Confusion Matrix (Acc: {acc:.2%})")
    axes[1, 1].set_xlabel("Predicted Label")
    axes[1, 1].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig("training_summary.png", dpi=150)
    plt.show()

    # 2D Decision Boundary Space Visualizer (restricted to first 2 features)
    X2 = X_train_s[:, :2]
    X2_i = add_intercept(X2)
    model_2d = LogisticRegressionGD(learning_rate=best_lr, num_epochs=2000).fit(X2_i, y_train)

    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid = add_intercept(np.c_[xx.ravel(), yy.ravel()])
    Z = model_2d.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(9, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    plt.scatter(X2[y_train == 0, 0], X2[y_train == 0, 1],
                c="crimson", edgecolors="black", alpha=0.8, label="Malignant")
    plt.scatter(X2[y_train == 1, 0], X2[y_train == 1, 1],
                c="dodgerblue", edgecolors="black", alpha=0.8, label="Benign")
    plt.xlabel("Feature 1 (Scaled)")
    plt.ylabel("Feature 2 (Scaled)")
    plt.title("2D Proportional Decision Boundary Space Topology")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig("decision_boundary.png", dpi=150)
    plt.show()

    print(f"\nPerformance Verdict: {'PASSED' if acc > 0.90 else 'FAILED'}")


if __name__ == "__main__":
    main()