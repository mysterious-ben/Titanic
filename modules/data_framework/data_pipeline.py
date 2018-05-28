"""
Data pipelines
"""

from typing import Union, Tuple, List
import numpy as np
import pandas as pd
import modules.data_framework.utildata as utdata


def _featuresPipeline(data: pd.DataFrame, sibSpCutoff: Union[None, int] = 1, parchCutoff: Union[None, int] = 1,
                      ageImputeMethod: str = 'mean', syntheticFeatures: bool = False) -> pd.DataFrame:
    """
    Data cleaning of features, the full pipeline

    Args
        data: DataFrame with features as columns
        sibSpCutoff: Level to clip SibSp feature
        parchCutoff: Level to clip Parch feature
        ageImputeMethod: Method used to impute Age feature: 'mean', 'median', 'mode', 'logistic', 'tree'
        syntheticFeatures: Add new synthetic features CabinVal and Title

    Returns
        DataFrame with transformed features as columns
    """

    dataC = data.copy()  # type: pd.DataFrame

    # -- Embarked, Sex
    assert 'male' in dataC.Sex.values
    assert 'female' in dataC.Sex.values
    assert 'S' in dataC.Embarked.values
    assert 'C' in dataC.Embarked.values
    assert 'Q' in dataC.Embarked.values
    utdata.imputeFeature(dataC, feature='Embarked', method='mode', verbose=False)
    dataC = pd.get_dummies(dataC, columns=['Embarked', 'Sex'], prefix_sep='', dtype=utdata.CATEGORICAL_TYPE)
    dataC.drop(columns=['EmbarkedQ', 'Sexmale'], inplace=True)
    dataC.rename(columns={'Sexfemale': 'Female'}, inplace=True)

    # -- Cabin, Name, Ticket
    if syntheticFeatures:
        dataC['CabinNan'] = dataC['Cabin'].isna().astype(utdata.CATEGORICAL_TYPE)
        dataC['AgeNan'] = dataC['Age'].isna().astype(utdata.CATEGORICAL_TYPE)
        titles = dataC.Name.apply(utdata.getTitle)
        titles[(titles == 'Mlle')] = 'Miss'
        titles[(titles == 'Mme')] = 'Mrs'
        titles[(titles != 'Mr') & (titles != 'Miss') & (titles != 'Mrs') & (titles != 'Master')] = 'Rare'
        dataC['Title'] = titles
        dataC = pd.get_dummies(dataC, columns=['Title'], prefix_sep='', dtype=utdata.CATEGORICAL_TYPE)
        dataC.drop(columns=['TitleMr', 'TitleMrs'], inplace=True)
    dataC.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)

    # -- Fare, Age
    utdata.clipFeature(dataC, 'Fare', nStd=3)
    utdata.clipFeature(dataC, 'Age', nStd=3)
    utdata.imputeFeature(dataC, feature='Fare', method='mean', verbose=False)
    utdata.imputeFeature(dataC, feature='Age', method=ageImputeMethod, methodValue=-100,
                         methodExclude=['Survived', 'EmbarkedS', 'EmbarkedC'],
                         verbose=False)
    assert dataC.isna().sum().sum() == 0, dataC.isna().sum()

    # -- SibSp, Parch
    if sibSpCutoff is not None: dataC.loc[dataC['SibSp'] > sibSpCutoff, 'SibSp'] = sibSpCutoff
    if parchCutoff is not None: dataC.loc[dataC['Parch'] > parchCutoff, 'Parch'] = parchCutoff

    # dataC.Survived = dataC.Survived.astype(CATEGORICAL_TYPE)

    return dataC


def featuresPipeline(data: pd.DataFrame, version: int = 1, verbose: bool=True):
    """
    Data cleaning of features, the full pipeline

    Args
        data: DataFrame with features as columns
        version: Version of the feature processing pipeline (1, 2, 3, 4, 5); version=5 is recommended
    """

    if verbose:
        print('-- Data pipeline v. {} --'.format(version), end='\n\n')
    if version == 1:
        return _featuresPipeline(data=data, sibSpCutoff=1, parchCutoff=1, ageImputeMethod='mean',
                                 syntheticFeatures=False)
    if version == 2:
        return _featuresPipeline(data=data, sibSpCutoff=2, parchCutoff=3, ageImputeMethod='mean',
                                 syntheticFeatures=False)
    if version == 3:
        return _featuresPipeline(data=data, sibSpCutoff=2, parchCutoff=3, ageImputeMethod='value',
                                 syntheticFeatures=False)
    if version == 4:
        return _featuresPipeline(data=data, sibSpCutoff=2, parchCutoff=3, ageImputeMethod='linear',
                                 syntheticFeatures=False)
    if version == 5:
        return _featuresPipeline(data=data, sibSpCutoff=2, parchCutoff=3, ageImputeMethod='tree',
                                 syntheticFeatures=False)
    if version == 6:
        return _featuresPipeline(data=data, sibSpCutoff=2, parchCutoff=3, ageImputeMethod='linear',
                                 syntheticFeatures=True)
    if version == 7:
        return _featuresPipeline(data=data, sibSpCutoff=2, parchCutoff=3, ageImputeMethod='tree',
                                 syntheticFeatures=True)
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


def featuresPipelineTrainTest(dataTrain: pd.DataFrame, dataTest: pd.DataFrame,
                              version: int = 1, verbose: bool=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combined data pipeline for train and test data, and indices of non-binary features"""

    data = _combineTrainTest(dataTrain=dataTrain, dataTest=dataTest, outcome='Survived')
    dataC = featuresPipeline(data=data, version=version, verbose=verbose)
    dataTrainC, dataTestC = _splitTrainTest(data=dataC, outcome='Survived')
    return dataTrainC, dataTestC


if __name__ == 'main':
    pass
