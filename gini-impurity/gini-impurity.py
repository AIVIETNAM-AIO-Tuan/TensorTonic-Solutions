import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    y_left = np.array(y_left)
    y_right = np.array(y_right)

    if len(y_left) == 0 and len(y_right) == 0:
      gini = float(0)
      return gini 
    val_1, freq_1 = np.unique(y_left, return_counts=True)
    val_2, freq_2 = np.unique(y_right, return_counts=True)
    
    
    n_left = np.sum(freq_1)
    n_right = np.sum(freq_2)
    n = n_left + n_right

    p_l = 0
    for i in freq_1:
      p_l+=(i/n_left)**2
    gini_left = 1 - p_l

    p_r = 0
    for i in freq_2:
      p_r += (i/n_right)**2
    
    gini_right = 1 - p_r

    gini = n_left/n*gini_left + n_right/n*gini_right
    return gini