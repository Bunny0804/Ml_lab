#Logistic Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv('/content/anger_treatment_data.csv')
X = np.asarray(data[['AngerTreatment']]).reshape(-1, 1)
Y = np.asarray(data['HeartAttack']).reshape(-1, 1)

def _sigmoid(z):
  return 1 / (1 + np.exp(-z))

def logistic_regression_fit(X, Y, learning_rate, n_iters, epsilon=1e-5):
  n_samples = X.shape[0]
  b1, b0 = 0.0, 0.0
  cost_history = []
  for _ in range(n_iters):
    z = X * b1 + b0
    y_pred = _sigmoid(z)
    log_loss = -(1 / n_samples) * np.sum(Y * np.log(y_pred + epsilon) + (1 - Y) * np.log(1 - y_pred + epsilon))
    cost_history.append(log_loss)
    db1 = -(1 / n_samples) * np.sum(X * (Y - y_pred))
    db0 = -(1 / n_samples) * np.sum(Y - y_pred)
    b1 -= learning_rate * db1
    b0 -= learning_rate * db0
    return b1, b0, cost_history

def accuracy(y_true, y_pred):
  return np.sum(y_true == y_pred) / len(y_true)
b1, b0, cost_history = logistic_regression_fit(X, Y, learning_rate=0.001, n_iters=190000)
y_pred_prob = _sigmoid(X * b1 + b0)
y_pred = np.where(y_pred_prob >= 0.5, 1, 0)
result = pd.DataFrame({
 'Actual': Y.flatten(),
 'Predicted': y_pred.flatten()
})
print(result)
print(f"Intercept (b0): {b0}")
print(f"Coefficient (b1): {b1}")
print(f"Accuracy: {accuracy(Y.flatten(), y_pred.flatten()):.2f}")
