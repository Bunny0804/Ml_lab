#And perceptron
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])
w = np.array([1,1])
theta=1.5
y_pred = []

def Heaviside(net):
    if net >= theta:
        return 1
    else:
        return 0

for i in range(X.shape[0]):
    x = X[i]
    wtsum= np.dot(x,w)
    pred = Heaviside(wtsum)
    print(x[0], " AND ",x[1]," -> ",pred)
    y_pred.append(pred) # append to y_pred array

slope = -w[0]/w[1]
intercept = (theta)/w[1]

x_plane = np.linspace(-2,2,10)
y_plane = slope * x_plane + intercept

plt.scatter(X[:,0], X[:,1], c=y)
plt.plot(x_plane, y_plane, '-')
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.show()
