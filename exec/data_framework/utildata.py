import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functools


def dropFeature(data, feature):
    if feature in data.columns:
        data.drop(columns=feature, inplace=True)
    else:
        print(feature, ' is not in the DF')


def imputeFeature(data, feature, method='mean'):
    if feature in data.columns:
        if method == 'mean':
            data[feature].fillna(data[feature].mean(), inplace=True)
        elif method == 'median':
            data[feature].fillna(data[feature].median(), inplace=True)
        elif method == 'mode':
            data[feature].fillna(data[feature].mode()[0], inplace=True)
        else:
            raise LookupError
    else:
        print(feature, ' is not in the DF')


def dummyFeature(data, feature, **dummiesKeywords):
    if feature in data.columns:
        dummyNa = data[feature].isna().any()
        return pd.get_dummies(data, columns=[feature], dummy_na=dummyNa, **dummiesKeywords)
    else:
        print(feature, ' is not in the DF')


def normFeatures(data, features=None, method='std'):
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


def clipFeature(data, feature, nStd=3):
    dMean = data[feature].mean()
    dStd = data[feature].std() * nStd
    data[feature].clip(lower=dMean - dStd, upper=dMean + dStd, inplace=True)


def subplotShape(n):
    nAxCol = int(np.ceil(np.sqrt(n)))
    nAxRow = nAxCol - 1 if nAxCol * (nAxCol - 1) >= n else nAxCol
    return nAxRow, nAxCol


def plotList(list, method, nList=None, fig=None, ax=None, nAxRow=None, nAxCol=None):
    nList = len(list) if nList is None else nList
    if fig is None:
        nAxRow, nAxCol = subplotShape(nList)
        fig, ax = plt.subplots(nrows=nAxRow, ncols=nAxCol)
    for c, xcol in enumerate(list):
        axIdx = np.unravel_index(c, (nAxRow, nAxCol))
        method(list[xcol], ax=ax[axIdx[0], axIdx[1]])
        # method(list[xcol], ax=axs[c])
    fig.tight_layout()


def histColumns(data, groupby=None):
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


def regplot(data, featX, featY):
    fig, ax = plt.subplots()
    sns.regplot(data[featX], data[featY], logistic=True, ax=ax)
    sns.regplot(data[featX], data[featY], lowess=True, ax=ax)
    fig.show()
