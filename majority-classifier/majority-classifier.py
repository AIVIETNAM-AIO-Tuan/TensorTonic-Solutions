import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    cls, count = np.unique(y_train, return_counts=True)

    pred = cls[np.argmax(count)]
        
    return np.full(len(X_test),pred).tolist()
    