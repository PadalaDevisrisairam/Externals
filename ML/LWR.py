import numpy as np
import matplotlib.pyplot as plt

def LWR(X_train, y_train, x_query, tau=1.0):
    m = X_train.shape[0]
    W = np.eye(m)
    for i in range(m):
        diff = X_train[i] - x_query
        W[i, i] = np.exp(-np.dot(diff, diff) / (2 * tau ** 2))
    
    X_design = np.c_[np.ones((m, 1)), X_train]
    theta = np.linalg.pinv(X_design.T @ W @ X_design) @ X_design.T @ W @ y_train
    x_query_with_bias = np.array([1, x_query])
    return x_query_with_bias @ theta

# Synthetic data
np.random.seed(42)
X = np.sort(np.random.rand(100, 1) * 10, axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.1

X_query = np.linspace(0, 10, 500)

# Tau values to compare
taus = [0.1, 0.3, 0.8, 2]

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', s=15, alpha=0.5, label='Training Data')

for tau in taus:
    y_pred = np.array([LWR(X, y, xq, tau=tau) for xq in X_query])
    plt.plot(X_query, y_pred, label=f'tau = {tau}')

plt.title("Locally Weighted Regression (Multiple Tau Values)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()