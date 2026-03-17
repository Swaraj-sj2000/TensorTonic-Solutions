import numpy as np

def _sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).reshape(-1)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    N, D = X.shape
    
    w = np.zeros(D)
    b = 0.0
    
    for _ in range(steps):
        z = X @ w + b
        p = _sigmoid(z)
        
        error = p - y
        
        dw = (X.T @ error) / N
        db = np.sum(error) / N
        
        w -= lr * dw
        b -= lr * db
    
    return w, float(b)