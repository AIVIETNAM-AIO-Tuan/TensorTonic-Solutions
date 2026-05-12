import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    x = np.array(X)
    y = np.array(y)
    N = x.shape[0]
    w = np.random.rand(x.shape[1])
    b = 0.0
    print(N)
    
    for _ in range(steps):
      #forward
      y_hat = x@w+b
      a = _sigmoid(y_hat)
      #backprop
      loss = a - y
      grad_w = (1/N)*np.transpose(x)@(a-y)
      grad_b = (1/N)*np.sum(a-y)
        
      w = w - lr*grad_w
      b = b - lr*grad_b

    return (w,b)
    

    