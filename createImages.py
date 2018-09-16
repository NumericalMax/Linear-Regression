import numpy as np
from matplotlib import pyplot as plt

params = np.loadtxt("./output.txt")
x = np.loadtxt("./X.txt")
y = np.loadtxt("./y.txt")

plt.figure(figsize=(20,10))
plt.scatter(x, y)
plt.plot(x, x*params[0]+params[1])
plt.savefig("./images/regressionLine.png")
