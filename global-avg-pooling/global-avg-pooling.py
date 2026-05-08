import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    x = np.array(x)
    if len(x.shape) == 4:
      N = x.shape[0]
      C = x.shape[1]
      H = x.shape[2]
      W = x.shape[3]
      print(N,C,H,W)
      gap = np.empty((N, C))
      for n in range(N):
        for c in range(C):
          gap_c = 0
          for h in range(H):
            for w in range(W):
                gap_c += x[n,c,h,w]
          gap_c = gap_c/(H*W)
          gap[n,c] =  gap_c
    elif len(x.shape) == 3:
      C = x.shape[0]
      H = x.shape[1]
      W = x.shape[2]
      print(C,H,W)
      gap = np.empty((C,)) 
      for c in range(C):
          gap_c = 0
          for h in range(H):
            for w in range(W):
                gap_c += x[c,h,w]
          gap_c = gap_c/(H*W)
          gap[c] =  gap_c
    else:
        raise ValueError
    return gap
    