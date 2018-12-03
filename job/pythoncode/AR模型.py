import numpy as np
import matplotlib.pyplot as plt
def AR(b, X, mu, sigma):
    """This functions simulates and autoregressive process
    by generating new values given historical values AR coeffs b1...bk + rand"""
    l = min(len(b) - 1, len(X))
    b0 = b[0]
 
    return b0 + np.dot(b[1:l + 1], X[-l:]) + np.random.normal(mu, sigma)
 
 
#Generate random data.
np.random.seed(8)
b = np.array([0.2, 0.04, 0.4, 0.05])
X = np.array([1])
mu = 0
sigma = 1
 
for i in range(1,1000):
    X = np.append(X, AR(b, X, mu, sigma))
 
#Plot the AR series.
fig, ax = plt.subplots(figsize = (15, 7))
plt.plot(X)
plt.xlabel("Time values")
plt.ylabel("AR values")
plt.show()
