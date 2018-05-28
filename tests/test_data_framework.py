import unittest
import modules.data_framework.data_pipeline as dtp

import numpy as np
import pandas as pd


class TestDataFramework(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.DataFrame(columns=['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
                                         'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], index=np.arange(1, 6))
        cls.data.Survived = [1, 1, 1, 0, 0]
        cls.data.Pclass = [3, 2, 1, 3, 3]
        cls.data.Name = ['Anna Frank', 'Anna Smith', 'Charles Gordon', 'Anna Drank', 'J A']
        cls.data.Sex = ['female', 'female', 'male', 'female', 'male']
        cls.data.Age = [20, np.nan, 666, 20, 23.2]
        cls.data.SibSp = [1, 10, 0, 1, 0]
        cls.data.Parch = [0, 15, 2, 0, 0]
        cls.data.Ticket = ['112312', '2232', 'CA. 21223', 'PC 4979070', '19234']
        cls.data.Fare = [13, 15.5, 666, 13, 8.2]
        cls.data.Cabin = ['E25', np.nan, np.nan, 'A12', np.nan]
        cls.data.Embarked = ['S', 'C', np.nan, 'Q', 'S']

        cls.dataC = pd.DataFrame(columns=['Survived', 'Pclass', 'Female', 'Age', 'SibSp',
                                          'Parch', 'Fare', 'EmbarkedS', 'EmbarkedC'], index=np.arange(1, 6))
        cls.dataC.Survived = [1, 1, 1, 0, 0]
        cls.dataC.Pclass = [3, 2, 1, 3, 3]
        cls.dataC.Female = [1, 1, 0, 1, 0]
        cls.dataC.Age = [20, np.nan, np.nan, 20, 23.2]
        cls.dataC.SibSp = [1, np.nan, 0, 1, 0]
        cls.dataC.Parch = [0, np.nan, np.nan, 0, 0]
        cls.dataC.Fare = [13, 15.5, np.nan, 13, 8.2]
        cls.dataC.EmbarkedS = [1, 0, 1, 0, 1]
        cls.dataC.EmbarkedC = [0, 1, 0, 0, 0]

        cls.dataT = pd.DataFrame(columns=['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
                                          'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], index=np.arange(6, 8))
        cls.dataT.Survived = [0, 1]
        cls.dataT.Pclass = [1, 2]
        cls.dataT.Name = ['Anna Crank', 'Jack France']
        cls.dataT.Sex = ['female', 'male']
        cls.dataT.Age = [20, np.nan]
        cls.dataT.SibSp = [1, 1]
        cls.dataT.Parch = [0, 1]
        cls.dataT.Ticket = ['112312', '2232']
        cls.dataT.Fare = [13, 9.5]
        cls.dataT.Cabin = [np.nan, 'B1']
        cls.dataT.Embarked = [np.nan, 'S']

    def test_featuresPipeline_version1(self):
        """
        Todo: Add a unit tests case for 3-std feature cap (?)
        Todo: Add unit tests for all versions of the data pipeline
        Todo: Add unit test for dtype=int8 check
        """

        version = 1
        data = dtp.featuresPipeline(data=self.data, version=version)

        self.assertSetEqual(set(data.columns.values.tolist()), set(self.dataC.columns.values.tolist()))
        np.testing.assert_equal(data.Survived.values, self.data.Survived.values)
        np.testing.assert_equal(data.Survived.values, self.dataC.Survived.values)
        pd.testing.assert_series_equal(data.Pclass, self.data.Pclass)
        pd.testing.assert_series_equal(data.Female, self.dataC.Female, check_dtype=False)
        pd.testing.assert_series_equal(data.EmbarkedS, self.dataC.EmbarkedS, check_dtype=False)
        pd.testing.assert_series_equal(data.EmbarkedC, self.dataC.EmbarkedC, check_dtype=False)
        np.testing.assert_equal(data.SibSp.values, [1, 1, 0, 1, 0])
        np.testing.assert_equal(data.Parch.values, [0, 1, 1, 0, 0])

        ageCap = np.median([20, 666, 20, 23.2]) + 3 * np.std([20, 666, 20, 23.2], ddof=1)
        age3 = min(666, ageCap)
        age2 = np.mean([20, age3, 20, 23.2])
        np.testing.assert_equal(data.Age.values, [20, age2, age3, 20, 23.2])

        fareCap = np.median([13, 15.5, 666, 13, 8.2]) + 3 * np.std([13, 15.5, 666, 13, 8.2], ddof=1)
        fare3 = min(666, fareCap)
        np.testing.assert_equal(data.Fare.values, [13, 15.5, fare3, 13, 8.2])

    def test_featuresPipelineTrainTest(self):
        for version in (1, 2, 3):
            with self.subTest(version=version):
                data, dataT = dtp.featuresPipelineTrainTest(dataTrain=self.data, dataTest=self.dataT, version=version)
                bmrk = dtp.featuresPipeline(data=self.data, version=version)

                np.testing.assert_equal(data.columns.values, bmrk.columns.values)
                pd.testing.assert_series_equal(data.Survived, bmrk.Survived)
                pd.testing.assert_series_equal(data.Pclass, bmrk.Pclass)
                pd.testing.assert_series_equal(data.Female, bmrk.Female)
                pd.testing.assert_series_equal(data.Survived, bmrk.Survived)
                pd.testing.assert_series_equal(data.EmbarkedS, bmrk.EmbarkedS)
                pd.testing.assert_series_equal(data.EmbarkedC, bmrk.EmbarkedC)
                pd.testing.assert_series_equal(data.SibSp, bmrk.SibSp)
                pd.testing.assert_series_equal(data.Parch, bmrk.Parch)

                ageCap = np.median([20, 666, 20, 23.2, 20]) + 3 * np.std([20, 666, 20, 23.2, 20], ddof=1)
                age3 = min(666, ageCap)
                if version == 1 or version == 2:
                    age2 = np.mean([20, age3, 20, 23.2, 20])
                elif version == 3:
                    age2 = -100.
                elif version == 4:
                    raise LookupError
                else:
                    raise LookupError
                np.testing.assert_equal(data.Age.values, [20, age2, age3, 20, 23.2])

                fareCap = np.median([13, 15.5, 666, 13, 8.2, 13, 9.5]) \
                          + 3 * np.std([13, 15.5, 666, 13, 8.2, 13, 9.5], ddof=1)
                fare3 = min(666, fareCap)
                np.testing.assert_equal(data.Fare.values, [13, 15.5, fare3, 13, 8.2])

                np.testing.assert_equal(dataT.columns.values, bmrk.columns.values)


if __name__ == '__main__':
    unittest.main()
