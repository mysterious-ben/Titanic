"""
Data pipelines
"""

import pandas as pd
import exec.data_framework.utildata as ut


def featuresPipeline(data: pd.DataFrame, version: int = 1) -> pd.DataFrame:
    """
    Data cleaning of features, the full pipeline (v. 1)

    Args
        data: DataFrame with features as columns
        version : version of pipeline

    Returns
        DataFrame with transformed features as columns
    """

    if version == 1:
        dataC = data.copy()
        dataC.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)

        ut.imputeFeature(dataC, 'Age', 'mean')
        ut.imputeFeature(dataC, 'Embarked', 'mode')

        assert dataC.isna().any().any() == False

        dataC = pd.get_dummies(dataC, columns=['Embarked', 'Sex'], prefix_sep='')
        dataC.drop(columns=['EmbarkedQ', 'Sexmale'], inplace=True)
        dataC.rename(columns={'Sexfemale': 'Female'}, inplace=True)

        dataC['SibSpYes'] = (dataC['SibSp'] > 0).astype(int)
        dataC['ParchYes'] = (dataC['Parch'] > 0).astype(int)
        dataC.drop(columns=['SibSp', 'Parch'], inplace=True)

        ut.clipFeature(dataC, 'Age', nStd=3)
        ut.clipFeature(dataC, 'Fare', nStd=3)
    else:
        raise LookupError

    return dataC


if __name__ == 'main':
    pass
