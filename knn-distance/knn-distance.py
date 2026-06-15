import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    x_train = np.array(X_train)
    x_test = np.array(X_test)

    if x_train.ndim == 1:
        x_train = x_train[:, np.newaxis]
    if x_test.ndim == 1:
        x_test = x_test[:, np.newaxis]

    dist = np.sqrt((x_test[:, np.newaxis, :] - x_train[np.newaxis, :, :])**2) 

    dist_aggregated = np.sum(dist, axis=-1)
    
    dist_sort = np.argsort(dist_aggregated, axis=1)

    n_test = x_test.shape[0]
    result = np.full((n_test,k),fill_value=-1)

    err = min(k, x_train.shape[0])
    
    result[:,:err] = dist_sort[:,:err]
 
    
    return result