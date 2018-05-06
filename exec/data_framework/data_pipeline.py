"""
Data pipelines
"""

from typing import Union
import numpy as  np
import pandas as pd
import exec.data_framework.utildata as utdata


def _featuresPipeline(data: pd.DataFrame, sibSpCutoff: Union[None, int] = 1, parchCutoff: Union[None, int] = 1,
                      ageImputeMethod: str = 'mean') -> pd.DataFrame:
    """
    Data cleaning of features, the full pipeline

    # Args
        data: DataFrame with features as columns
        sibSpCutoff: Level to clip SibSp feature
        parchCutoff: Level to clip Parch feature
        ageImputeMethod: Method used to impute Age feature ('mean', 'median', 'mode', 'logistic', 'tree')

    Returns
        DataFrame with transformed features as columns
    """

    dataC = data.copy()  # type: pd.DataFrame
    dataC.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)

    utdata.imputeFeature(dataC, feature='Embarked', method='mode')

    assert 'male' in dataC.Sex.values
    assert 'female' in dataC.Sex.values
    assert 'S' in dataC.Embarked.values
    assert 'C' in dataC.Embarked.values
    assert 'Q' in dataC.Embarked.values
    dataC = pd.get_dummies(dataC, columns=['Embarked', 'Sex'], prefix_sep='')
    dataC.drop(columns=['EmbarkedQ', 'Sexmale'], inplace=True)
    dataC.rename(columns={'Sexfemale': 'Female'}, inplace=True)

    utdata.clipFeature(dataC, 'Fare', nStd=3)
    utdata.clipFeature(dataC, 'Age', nStd=3)

    utdata.imputeFeature(dataC, feature='Fare', method='mean')
    utdata.imputeFeature(dataC, feature='Age', method=ageImputeMethod, methodValue=-100, methodExclude=['Survived'])
    assert dataC.isna().sum().sum() == 0, dataC.isna().sum()

    if sibSpCutoff is not None: dataC.loc[dataC['SibSp'] > sibSpCutoff, 'SibSp'] = sibSpCutoff
    if parchCutoff is not None: dataC.loc[dataC['Parch'] > parchCutoff, 'Parch'] = parchCutoff

    return dataC


def featuresPipeline(data: pd.DataFrame, version: int = 1):
    """
    Data cleaning of features, the full pipeline

    Args
        data: DataFrame with features as columns
        version: Version of the feature processing pipeline (1, 2, 3, 4, 5); version=5 is recommended

    Todo: Add a check for negative Age
    """

    if version == 1:
        return _featuresPipeline(data=data, sibSpCutoff=1, parchCutoff=1, ageImputeMethod='mean')
    if version == 2:
        return _featuresPipeline(data=data, sibSpCutoff=2, parchCutoff=3, ageImputeMethod='mean')
    if version == 3:
        return _featuresPipeline(data=data, sibSpCutoff=2, parchCutoff=3, ageImputeMethod='value')
    if version == 4:
        return _featuresPipeline(data=data, sibSpCutoff=2, parchCutoff=3, ageImputeMethod='logistic')
    if version == 5:
        return _featuresPipeline(data=data, sibSpCutoff=2, parchCutoff=3, ageImputeMethod='tree')
    else:
        raise LookupError


def _combineTrainTest(dataTrain: pd.DataFrame, dataTest: pd.DataFrame, outcome: str = 'Survived'):
    # dataTest[outcome] = -1
    np.testing.assert_equal(dataTrain.columns.values, dataTest.columns.values)
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
