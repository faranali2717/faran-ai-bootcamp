"""
DAY 2: Calculus for AI - Chain Rule & Partial Derivatives
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("DAY 2: CALCULUS FOR AI")
print("=" * 50)

# ----------------------------------------------
# PART 1: Chain Rule - Analytical vs Numerical
# ----------------------------------------------

print("\n--- Chain Rule: Derivative of (3x² + x)³ ---")

def f(x):
    """Original function: (3x² + x)³"""
    return (3*x**2 + x)**3

def df_analytical(x):
    """Derivative we calculated by hand: 3(3x² + x)²(6x + 1)"""
    return 3 * (3*x**2 + x)**2 * (6*x + 1)

def df_numerical(x, h=1e-6):
    """Approximate derivative using difference quotient"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Test at x = 2
x_test = 2
analytical = df_analytical(x_test)
numerical = df_numerical(x_test)

print(f"At x = {x_test}")
print(f"  Analytical derivative: {analytical:.2f}")
print(f"  Numerical derivative:  {numerical:.2f}")
print(f"  Match? {abs(analytical - numerical) < 0.001} ✅")

# Plot the function and its derivative
x_vals = np.linspace(-2, 1, 100)
y_vals = f(x_vals)
dy_vals = df_analytical(x_vals)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, 'b-', linewidth=2)
plt.title("f(x) = (3x² + x)³")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_vals, dy_vals, 'r-', linewidth=2)
plt.title("f'(x) = 3(3x² + x)²(6x + 1)")
plt.grid(True)
plt.tight_layout()
plt.savefig('day2_chain_rule.png')
print("\n Saved as 'day2_chain_rule.png'")

# ----------------------------------------------
# PART 2: Partial Derivative - With Respect to x
# ----------------------------------------------

print("\n--- Partial Derivative: ∂/∂x of (x²y + y²) ---")

def partial_derivative_x(x, y, h=1e-6):
    """Numerical partial derivative with respect to x"""
    def g(x, y):
        return x**2 * y + y**2
    
    return (g(x + h, y) - g(x - h, y)) / (2 * h)

# Test at x = 3, y = 2
x_test, y_test = 3, 2
analytical_partial = 2 * x_test * y_test  # Our answer: 2xy
numerical_partial = partial_derivative_x(x_test, y_test)

print(f"At (x, y) = ({x_test}, {y_test})")
print(f"  Analytical ∂/∂x: {analytical_partial}")
print(f"  Numerical ∂/∂x:  {numerical_partial:.6f}")
print(f"  Match? {abs(analytical_partial - numerical_partial) < 0.001} ✅")

