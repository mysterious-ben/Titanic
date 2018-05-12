from typing import Sequence, Iterable, Union
import numpy as np
import pandas as pd
from sklearn import linear_model as sklm
from sklearn import ensemble as skens
import matplotlib.pyplot as plt
import seaborn as sns
import functools
import itertools
import re


def dropFeature(data: pd.DataFrame, feature: str, inplace=True) -> Union[None, pd.DataFrame]:
    if feature in data.columns:
        return data.drop(columns=feature, inplace=inplace)
    else:
        print(feature, ' is not in the DF')


def dropFeatures(data: pd.DataFrame, features: Iterable[str], inplace=True) -> Union[None, pd.DataFrame]:
    featuresC = (x for x in features if x in data.columns)
    return data.drop(columns=featuresC, inplace=inplace)


def imputeFeature(data: pd.DataFrame, feature: str, method: str = 'mean',
                  methodValue: float = None, methodExclude: Sequence = tuple()) -> None:
    if feature in data.columns:
        if data[feature].isna().sum() == 0:
            print('No NaNs')
            return
        if method == 'mean':
            data[feature].fillna(data[feature].mean(), inplace=True)
        elif method == 'median':
            data[feature].fillna(data[feature].median(), inplace=True)
        elif method == 'mode':
            data[feature].fillna(data[feature].mode()[0], inplace=True)
        elif method == 'value':
            data[feature].fillna(methodValue, inplace=True)
        elif method == 'linear':
            regr = sklm.LinearRegression(fit_intercept=True)
            X = data.drop(columns=itertools.chain(methodExclude, [feature]))  # type: pd.DataFrame
            y = data[feature]
            regr.fit(X=X.loc[y.notna(), :], y=y.loc[y.notna()])
            data.loc[y.isna(), feature] = regr.predict(X=X.loc[y.isna(), :])
        elif method == 'tree':
            regr = skens.RandomForestRegressor(n_estimators=64, max_leaf_nodes=16, random_state=1)
            X = data.drop(columns=itertools.chain(methodExclude, [feature]))  # type: pd.DataFrame
            y = data[feature]
            regr.fit(X=X.loc[y.notna(), :], y=y.loc[y.notna()])
            data.loc[y.isna(), feature] = regr.predict(X=X.loc[y.isna(), :])
        else:
            raise LookupError
    else:
        print(feature, ' is not in the DF')


def dummyFeature(data: pd.DataFrame, feature: str, **dummiesKeywords) -> pd.DataFrame:
    if feature in data.columns:
        dummyNa = data[feature].isna().any()
        return pd.get_dummies(data, columns=[feature], dummy_na=dummyNa, **dummiesKeywords)
    else:
        print(feature, ' is not in the DF')


def normFeatures(data, features=None, method='std') -> pd.DataFrame:
    Y = data.copy()
    if features is None: features = Y.columns
    if method == 'std':
        Y[features] = (Y[features] / Y[features].std(axis=0))
    elif method == 'stdmean':
        Y[features] = (Y[features] - Y[features].mean(axis=0)) / Y[features].std(axis=0)
    elif method == 'maxmin':
        Y[features] = (Y[features] - Y[features].min(axis=0)) / (Y[features].max(axis=0) - Y[features].min(axis=0))
    else:
        raise LookupError
    return Y


def clipFeature(data: pd.DataFrame, feature: str, nStd=3) -> None:
    dMean = data[feature].median()
    dStd = data[feature].std() * nStd
    data[feature].clip(lower=dMean - dStd, upper=dMean + dStd, inplace=True)


def subplotShape(n: int):
    nAxCol = int(np.ceil(np.sqrt(n)))
    nAxRow = nAxCol - 1 if nAxCol * (nAxCol - 1) >= n else nAxCol
    return nAxRow, nAxCol


def plotList(list, method, nList=None, fig=None, ax=None, nAxRow=None, nAxCol=None) -> None:
    nList = len(list) if nList is None else nList
    if fig is None:
        nAxRow, nAxCol = subplotShape(nList)
        fig, ax = plt.subplots(nrows=nAxRow, ncols=nAxCol)
    for c, xcol in enumerate(list):
        axIdx = np.unravel_index(c, (nAxRow, nAxCol))
        method(list[xcol], ax=ax[axIdx[0], axIdx[1]])
        # method(list[xcol], ax=axs[c])
    fig.tight_layout()


def histColumns(data: pd.DataFrame, groupby=None) -> None:
    if groupby is None:
        plotList(list=data,
                 method=functools.partial(sns.distplot, kde=False,
                                          hist_kws={"rwidth": 0.5, 'edgecolor': 'black', 'alpha': 1.0}),
                 nList=data.shape[1])
    else:
        nList = data.shape[1]
        nAxRow, nAxCol = subplotShape(nList)
        fig, ax = plt.subplots(nrows=nAxRow, ncols=nAxCol)
        for x in data.groupby(groupby):
            plotList(list=x[1],
                     method=functools.partial(sns.distplot, kde=False,
                                              hist_kws={"rwidth": 0.5, 'edgecolor': 'black', 'alpha': 0.5}),
                     nList=nList,
                     fig=fig, ax=ax, nAxRow=nAxRow, nAxCol=nAxCol)


def regplot(data: pd.DataFrame, featX: str, featY: str) -> None:
    fig, ax = plt.subplots()
    sns.regplot(data[featX], data[featY], logistic=True, ax=ax)
    sns.regplot(data[featX], data[featY], lowess=True, ax=ax)
    fig.show()


def getTitle(name: str) -> str:
    match = re.match('.*, ([\w\s]*)\. .*', name)
    if match is None:
        return ''
    else:
        return match.groups(0)[0]


if __name__ == '__main__':
    print(getTitle('Moor, Master. Meier'))