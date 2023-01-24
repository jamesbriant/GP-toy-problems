import numpy as np
import matplotlib.pyplot as plt

# x = [1.2, 1.5, 1.9]
# x = [1.01, 1.1, 1.15, 1.2, 1.3, 1.35, 1.5, 1.62, 1.7, 1.8, 1.9, 1.99]
# f = lambda x: 1 + (np.sin(3*x) + np.sin(10*x) + np.sin(30*x))/3

x = [0.1, 0.4, 0.7, 1, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7]
f = lambda x : 1.5 + 0.1*x**3 - x

y = [f(xi) for xi in x]

minimum = 1
x1 = 0
x2 = 0

for i in range(len(y)):
    for j in range(len(y) - i - 1):
        a = np.abs(y[i] - y[j+i+1])
        if a < minimum:
            minimum = a
            x1 = x[i]
            x2 = x[i+j+1]
        # print(x[i], x[i+j+1], a)

print(x1, x2, minimum)

plt.plot(x, y)
plt.show()