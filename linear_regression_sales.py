# linear_regression_sales.py

"""
Linear Regression from Scratch: Predict Sales based on Number of Customers

Author: sudeshna
Date: 2025-12-15

This script:
- Uses a simple dataset of number of customers vs sales
- Implements Linear Regression from scratch using gradient descent
- Plots the regression line
- Can be uploaded to GitHub
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Sample Data
# -------------------------------
# Number of customers (independent variable X)
customers = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# Corresponding sales in $ (dependent variable Y)
sales = np.array([15, 25, 35, 45, 50, 65, 70, 80, 95, 105])

# Normalize data (optional, helps gradient descent)
X = customers
Y = sales

# -------------------------------
# Step 2: Initialize Parameters
# -------------------------------
m = 0  # slope
b = 0  # intercept
learning_rate = 0.0005
epochs = 1000
n = float(len(X))

# -------------------------------
# Step 3: Gradient Descent
# -------------------------------
for i in range(epochs):
    Y_pred = m * X + b
    # Compute gradients
    D_m = (-2/n) * sum(X * (Y - Y_pred))
    D_b = (-2/n) * sum(Y - Y_pred)
    # Update parameters
    m = m - learning_rate * D_m
    b = b - learning_rate * D_b
    
    # Optional: print loss every 100 iterations
    if i % 100 == 0:
        loss = sum((Y - Y_pred) ** 2) / n
        print(f"Epoch {i}, Loss: {loss:.2f}, m: {m:.2f}, b: {b:.2f}")

# -------------------------------
# Step 4: Final Model
# -------------------------------
print(f"\nFinal model: Sales = {m:.2f} * Customers + {b:.2f}")

# -------------------------------
# Step 5: Visualization
# -------------------------------
plt.scatter(X, Y, color='blue', label='Actual Sales')
plt.plot(X, m*X + b, color='red', label='Regression Line')
plt.xlabel('Number of Customers')
plt.ylabel('Sales ($)')
plt.title('Linear Regression: Customers vs Sales')
plt.legend()
plt.show()

# -------------------------------
# Step 6: Predict Example
# -------------------------------
new_customers = 120
predicted_sales = m * new_customers + b
print(f"Predicted sales for {new_customers} customers: ${predicted_sales:.2f}")
