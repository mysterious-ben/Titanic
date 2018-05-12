import unittest
import modules.data_framework.utildata as utdata
import modules.model_framework.utilmodel as utmdl

import numpy as np
import pandas as pd


class TestUtil(unittest.TestCase):
    """
    Unit tests for utility modules
    """

    @classmethod
    def setUpClass(cls):
        cls.EPS = 1e-7

        cls.cols = ['a', 'b', 'c']

        cls.arrays = {'full': np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 3, 3]]),
                      'nan1': np.array([[0, 0, 0], [0, 0, 1], [np.nan, 0, 2], [np.nan, 3, 3]]),
                      'nan2': np.array([[0, 0, 0], [0, 0, 1], [0, np.nan, 2], [0, 3, 3]]),
                      'nan3': np.array([[0, 0, 0], [0, np.nan, 1], [0, np.nan, 2], [0, 3, 3]])}
        cls.dfs = {x: pd.DataFrame(y, columns=cls.cols) for x, y in cls.arrays.items()}

    def test_imputeFeature_trivial_mean_median_mode(self):
        for m in ['mean', 'median', 'mode']:
            with self.subTest(method=m):
                data = self.dfs['nan1'].copy()
                utdata.imputeFeature(data=data, feature='a', method=m)
                pd.testing.assert_frame_equal(data, self.dfs['full'], check_dtype=False)

    def test_imputeFeature_mean(self):
        data = self.dfs['nan2'].copy()
        utdata.imputeFeature(data=data, feature='b', method='mean')
        self.assertAlmostEqual(data.loc[2, 'b'], 1.)

    def test_imputeFeature_median(self):
        data = self.dfs['nan2'].copy()
        utdata.imputeFeature(data=data, feature='b', method='median')
        self.assertAlmostEqual(data.loc[2, 'b'], 0.)

    def test_imputeFeature_value(self):
        data = self.dfs['nan2'].copy()
        utdata.imputeFeature(data=data, feature='b', method='value', methodValue=-5)
        self.assertAlmostEqual(data.loc[2, 'b'], -5)

    def test_imputeFeature_regress(self):
        data = self.dfs['nan3'].copy()
        utdata.imputeFeature(data=data, feature='b', method='linear', methodExclude=['a'])
        self.assertAlmostEqual(data.loc[1, 'b'], 1)
        self.assertAlmostEqual(data.loc[2, 'b'], 2)

    def test_clipFeature_noclip(self):
        data = self.dfs['full'].copy()
        utdata.clipFeature(data=data, feature='c', nStd=3)
        pd.testing.assert_frame_equal(data, self.dfs['full'], check_dtype=False)

    def test_clipFeature_clip(self):
        data = self.dfs['full'].copy()
        utdata.clipFeature(data=data, feature='b', nStd=0.5)
        self.assertAlmostEqual(data.loc[3, 'b'], 0.0 + 0.5 * np.std([0, 0, 0, 3], ddof=1))

    def test_getTitle_match(self):
        self.assertEqual(utdata.getTitle('Moor, Master. Meier'), 'Master')

    def test_getTitle_nomatch(self):
        self.assertEqual(utdata.getTitle('Moor, Master; Meier'), '')

    def test_dataframeToXy(self):
        data = self.dfs['full'].copy()
        X, y = utmdl.dataframeToXy(data=data, outcome='c')
        pd.testing.assert_frame_equal(X, data.drop(columns=['c']), check_dtype=False)
        np.testing.assert_array_equal(y, data['c'].values)

    def test_proba2d(self):
        proba1d = np.array([1, 0.5, 0, 0])
        proba2d = np.array([[0, 1], [0.5, 0.5], [1, 0], [1, 0]])
        np.testing.assert_equal(utmdl.proba2d(proba1d=proba1d), proba2d)

    def test_weightsGrid(self):
        grid = [[0.0, 0.0, 1.0], [0.0, 0.5, 0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]
        np.testing.assert_equal(utmdl.weightsGrid(n=3, step=0.5), grid)

    def test_deviance_equal_weights(self):
        np.testing.assert_equal(utmdl.deviance(w=np.array([0.5, 0.5]), X=np.array([[0, 0], [1, 1], [0, 0.5]])-0.5,
                                               y=np.array([0, 1, 1])-0.5, probaEps=0.1),
                                -2 * (np.log(0.9) + np.log(0.9) + np.log(0.25)))

    def test_deviance_diff_weights(self):
        np.testing.assert_equal(utmdl.deviance(w=np.array([0, 1]), X=np.array([[0.3, 0], [0, 1], [0, 0.5]])-0.5,
                                               y=np.array([0, 1, 1])-0.5, probaEps=0.1),
                                -2 * (np.log(0.9) + np.log(0.9) + np.log(0.5)))


if __name__ == '__main__':
    unittest.main()
