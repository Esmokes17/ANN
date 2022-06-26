import pandas as pd
import numpy as np

def loadCsv(filename: str):
    data = pd.read_csv(filename)
    X, targets = data.values[:,0:], data.values[:,0]
    
    Y = []
    for a in targets:
        row = [0] * 10
        row[a] = 1
        Y.append(row)

    return X, np.array(Y)
