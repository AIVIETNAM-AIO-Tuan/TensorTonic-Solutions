import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)
    if rng is None:
        prob = (np.random.random(x.shape) < (1 - p))
    else:
        prob = (rng.random(x.shape) < (1 - p))

  
    scaling = 1/(1-p)
    dropout_pattern = ()
    x_new = x*prob*scaling
    
    if p !=0:
      dropout_pattern = (prob)*(1/(1-p))
    else:
      dropout_pattern = (prob)*(1/(1-p))
    
    
    return x_new, dropout_pattern
