import numpy as np
import pandas as pd
import itertools


def dataframeToXy(data: pd.DataFrame, outcome='Survived'):
    """Split DataFrame to X and y (features and outcome)"""
    if outcome in data.columns:
        X = data.drop(columns=[outcome])
        y = data[outcome].values
    else:
        X = data.copy()
        y = None
    return X, y


def proba2d(proba1d: np.ndarray) -> np.ndarray:
    """Transform 1-d probability array to 2-d"""
    proba2d = np.zeros((proba1d.shape[0], 2), dtype=np.float)
    proba2d[:, 1] = proba1d
    proba2d[:, 0] = 1 - proba2d[:, 1]
    return proba2d


def weightsGrid(n, step=0.1):
    """Generate weight grid"""
    return [[*x, np.max(1 - np.sum(x), 0)] for x in
            itertools.product(*itertools.repeat(np.arange(0, 1 + step / 2., step), times=n - 1))
            if np.sum(x) < 1 + step / 2]


def deviance(w, X, y, probaEps=1e-3):
    """Deviance loss"""
    p = np.dot(X, w)
    p[p < -0.5 + probaEps] = -0.5 + probaEps
    p[p > 0.5 - probaEps] = 0.5 - probaEps
    return -2 * np.sum((0.5 + y) * np.log(0.5 + p) + (0.5 - y) * np.log(0.5 - p))


def squareH(w, X, y, threshold = 0.5):
    """Hubert square loss"""
    # X, y = (X - 0.5), (y - 0.5)
    p = np.dot(X, w)
    a = np.abs(y - p)
    return np.sum((a > threshold) * a + (a <= threshold) * a ** 2)


if __name__ == '__main__':
    print(weightsGrid(3, 0.5))
