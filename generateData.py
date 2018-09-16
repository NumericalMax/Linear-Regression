import numpy as np
from matplotlib import pyplot as plt

n = 100
x = range(0,n)
y = range(0,n)
for k in range(0, n):
    y[k] = y[k] + 3*np.random.randn() + 100

plt.figure(figsize=(20,10))
plt.scatter(x, y)
plt.savefig("./images/rawData.png")

X = np.zeros([n,1])
target = np.zeros([n,1])
X[:,0] = x
target[:,0] = y
np.savetxt("X.txt", X, delimiter=",", fmt='%f')
np.savetxt("y.txt", target, delimiter=",", fmt='%f')
