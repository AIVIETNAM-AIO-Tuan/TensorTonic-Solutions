import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.array(x)
    p = np.array(p)
    
    y = np.sum(p)
    if int(y) == 1:
        return float(np.dot(x, p)) 
    else:
        raise ValueError
