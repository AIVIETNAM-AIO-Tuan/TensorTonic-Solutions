import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """

    x = np.array(x)
    w = np.array(W)

    #get mat size
    N = x.shape[0]
    C_in = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]

    #get kernel size
    C_out = w.shape[0]
    KH = w.shape[2]
    KW = w.shape[3]
    
    H_out = H - KH + 1
    W_out = W - KW + 1
    y = [[[[0 for _ in range(W_out)]for _ in range(H_out)]for _ in range(C_out)]for _ in range(N)]
    
    
    y = np.array(y, dtype=float)
    for n in range(N):
        for c in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    total = np.sum(x[n,:,i:i+KH,j:j+KW] * w[c,:,:,:])
                    total = float(total)    
                    y[n,c,i,j] = float(total + b[c])
    return y
    
                