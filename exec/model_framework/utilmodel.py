import numpy as np

def dataframeToXy(data):
    if 'Survived' in data.columns:
        X = data.drop(columns=['Survived'])
        y = data['Survived'].values
    else:
        X = data.copy()
        y = None
    return X, y

def proba2d(proba1d: np.ndarray) -> np.ndarray:
    proba2d = np.zeros((proba1d.shape[0], 2), dtype=np.float)
    proba2d[:, 1] = proba1d
    proba2d[:, 0] = 1 - proba2d[:, 1]
    return proba2d