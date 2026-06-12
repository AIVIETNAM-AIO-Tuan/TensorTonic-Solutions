import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    cls, count = np.unique(y_train, return_counts=True)

    id = np.where(count == np.max(count))
    id = int(id[0][0])
    pred = cls[id]
    
    output = []
    for i in range(len(X_test)):
      output.append(pred)
    
    return output
    