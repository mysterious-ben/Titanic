"""
Data pipelines
"""

from typing import Union
import numpy as  np
import pandas as pd
import exec.data_framework.utildata as ut


def _featuresPipeline(data: pd.DataFrame, sibSpCutoff: Union[None, int] = 1, parchCutoff: Union[None, int] = 1,
                      ageImputeMethod: str = 'mean') -> pd.DataFrame:
    """
    Data cleaning of features, the full pipeline

    # Args
        data: DataFrame with features as columns
        SibSpCutoff:
        ParchCutoff:
        AgeImpute:

    Returns
        DataFrame with transformed features as columns
    """

    dataC = data.copy()  # type: pd.DataFrame
    dataC.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)

    ut.imputeFeature(dataC, feature='Embarked', method='mode')

    dataC = pd.get_dummies(dataC, columns=['Embarked', 'Sex'], prefix_sep='')
    dataC.drop(columns=['EmbarkedQ', 'Sexmale'], inplace=True)
    dataC.rename(columns={'Sexfemale': 'Female'}, inplace=True)

    ut.imputeFeature(dataC, feature='Age', method=ageImputeMethod, methodValue=-100, methodExclude=['Survived'])
    assert dataC.isna().any().any() == False

    if sibSpCutoff is not None: dataC.loc[dataC['SibSp'] > sibSpCutoff, 'SibSp'] = sibSpCutoff
    if parchCutoff is not None: dataC.loc[dataC['Parch'] > parchCutoff, 'Parch'] = parchCutoff

    ut.clipFeature(dataC, 'Fare', nStd=3)
    ut.clipFeature(dataC, 'Age', nStd=3)

    return dataC


def featuresPipeline(data: pd.DataFrame, version: int = 1):
    """
    Data cleaning of features, the full pipeline

    Args
        data: DataFrame with features as columns
        version:
    """

    if version == 1:
        return _featuresPipeline(data=data, sibSpCutoff=1, parchCutoff=1, ageImputeMethod='mean')
    if version == 2:
        return _featuresPipeline(data=data, sibSpCutoff=2, parchCutoff=3, ageImputeMethod='mean')
    if version == 3:
        return _featuresPipeline(data=data, sibSpCutoff=2, parchCutoff=3, ageImputeMethod='value')
    if version == 4:
        return _featuresPipeline(data=data, sibSpCutoff=2, parchCutoff=3, ageImputeMethod='regress')
    else:
        raise LookupError


def _combineTrainTest(dataTrain: pd.DataFrame, dataTest: pd.DataFrame, outcome: str = 'Survived'):
    # dataTest[outcome] = -1
    assert all(dataTrain.columns.values == dataTest.columns.values)
    return pd.concat([dataTrain, dataTest], axis=0, join='inner', keys=[False, True])


def _splitTrainTest(data: pd.DataFrame, outcome: str = 'Survived'):
    dataTrain = data.loc[False, :].copy()
    dataTest = data.loc[True, :].copy()
    # dataTest.drop(columns=[outcome], inplace=True)
    return dataTrain, dataTest


def featuresPipelineTrainTest(dataTrain: pd.DataFrame, dataTest: pd.DataFrame, version: int = 1):
    data = _combineTrainTest(dataTrain=dataTrain, dataTest=dataTest, outcome='Survived')
    dataC = featuresPipeline(data=data, version=version)
    return _splitTrainTest(data=dataC, outcome='Survived')


if __name__ == 'main':
    pass
