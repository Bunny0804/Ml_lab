#Gradient descent 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Input Data
X = np.asarray([20, 5, 10, 13, 12])
Y = np.asarray([24, 22, 25, 32, 27])

# Initialize Parameters
b1, b0 = 0, 0

# Learning Rate and Number of Iterations
L = 0.001
epochs = 20
n = np.size(X)

# For Gradient Descent Visualization
b1_history, loss_history = [], []

# Perform Gradient Descent
for i in range(epochs):
    Y_pred = b1 * X + b0
    loss = (1 / n) * sum((Y - Y_pred) ** 2)  # Mean Squared Error
    b1_history.append(b1)
    loss_history.append(loss)

    D_b1 = (-2 / n) * sum(X * (Y - Y_pred))  # Derivative wrt b1
    D_b0 = (-2 / n) * sum(Y - Y_pred)       # Derivative wrt b0

    b1 -= L * D_b1  # Update b1
    b0 -= L * D_b0  # Update b0

# Final Parameters
print(f"Final coefficients: b1={b1}, b0={b0}")

# Predicted Values
Y_pred = b1 * X + b0
mse = mean_squared_error(Y, Y_pred)
cd = r2_score(Y, Y_pred)
print('MSE:', mse)
print('Coefficient of determination:', cd)

# Visualization 1: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.plot(range(n), Y, 'o', label='Actual', color='blue')
plt.plot(range(n), Y_pred, 'x-', label='Predicted', color='red')
plt.title('Actual vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()

# Visualization 2: Gradient Descent Progression
plt.figure(figsize=(10, 8))
plt.plot(range(epochs), loss_history, label='Loss')
plt.title('Gradient Descent Progression')
plt.xlabel('Iteration')
plt.ylabel('Loss (Mean Squared Error)')
plt.grid()
plt.legend()
plt.show()
