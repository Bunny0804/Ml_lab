# Or gate pereptron
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])
w = np.array([1,1])
theta = 1
y_pred = []

def Heaviside(net):
    if net >= theta:
        return 1
    else:
        return 0

for i in range(X.shape[0]):
    x = X[i]
    wtsum = np.dot(x,w)
    pred = Heaviside(wtsum)
    print(x[0], "OR", x[1], "->", pred)
    y_pred.append(pred)

import matplotlib.pyplot as plt

slope = -w[0]/w[1]
intercept = (theta)/w[1]

x_plane = np.linspace(-2,2,10)
y_plane = slope * x_plane + intercept

plt.scatter(X[:,0], X[:,1], c=y)
plt.plot(x_plane-0.02, y_plane-0.02, '-')
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.show()
