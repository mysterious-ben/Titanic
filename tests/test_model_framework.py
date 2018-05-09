import unittest
import modules.model_framework.model as mdl

import numpy as np
import pandas as pd


class TestMetrics(unittest.TestCase):
    """
    Todo: Add unit tests for other metrics
    """

    def test_generator_accuracy(self):
        metric = mdl.Metrics.generator('accuracy')
        self.assertFalse(metric[1])
        self.assertAlmostEqual(metric[0]([1, 0, 0, 1, 0], [1, 0, 0, 1, 1]), 0.80)
        self.assertAlmostEqual(metric[0]([1, 0, 0, 1, 0], [0, 1, 0, 1, 1]), 0.40)


class TestModel(unittest.TestCase):
    """
    Todo: Add unit tests for other classification models
    """

    @classmethod
    def setUpClass(cls):
        cls.data1 = pd.DataFrame([[-1, 0, 0], [1, 0, 1], [-2.3, 0, 0], [3, 0, 1]], columns=['a', 'b', 'Survived'],
                                 index=np.arange(1, 5))
        cls.dataT1 = pd.DataFrame([[-1, 10, 0], [11.2, 0, 1]], columns=['a', 'b', 'Survived'], index=np.arange(5, 7))
        cls.data2 = pd.DataFrame([[-1, 0, 0], [1, 0, 1], [0, -1, 1], [0, 1, 0]], columns=['a', 'b', 'Survived'],
                                 index=np.arange(1, 5))
        cls.dataT2 = pd.DataFrame([[10, 1, 1], [1, 10, 0]], columns=['a', 'b', 'Survived'], index=np.arange(5, 7))
        cls.data3 = pd.DataFrame([[-5, 0], [-4, 0], [-3, 0], [-2, 0], [-0.5, 0],
                                  [0, 1], [1, 1], [2, 1], [3, 1], [4, 1]],
                                 columns=['a', 'Survived'], index=np.arange(1, 11))
        cls.dataT3 = pd.DataFrame([[-10, 0], [-4, 0], [-3, 0], [-2, 0], [-1, 0],
                                   [0, 1], [1, 1], [2, 1], [3, 1], [10, 1]],
                                  columns=['a', 'Survived'], index=np.arange(11, 21))

    def test_Logistic_prediction_1(self):
        model = mdl.Logistic(scale=True, fit_intercept=False)
        model.fit(data=self.data1)
        model.predict(data=self.dataT1)
        self.assertGreater(model._getBaseClassifier().coef_[0][0], 0)
        self.assertAlmostEqual(model._getBaseClassifier().coef_[0][1], 0)
        np.testing.assert_equal(model.ytH, model.yt)

    def test_Logistic_prediction_2(self):
        model = mdl.Logistic(scale=True, fit_intercept=False)
        model.fit(data=self.data2)
        model.predict(data=self.dataT2)
        self.assertGreater(model._getBaseClassifier().coef_[0][0], 0)
        self.assertLess(model._getBaseClassifier().coef_[0][1], 0)
        np.testing.assert_equal(model.ytH, model.yt)

    def test_genModelCV_benchmark_KNNCV(self):
        model = mdl.genModelCV(ModelClass=mdl.KNN, grid={'n_neighbors': [1, 3]}) \
            (cv=2, scale=True, weights='uniform')
        model.fit(data=self.data3)
        model.predict(data=self.dataT3)
        bmrk = mdl.KNNCV(cv=2, scale=True, weights='uniform', grid={'n_neighbors': np.arange(1, 3)})
        bmrk.fit(data=self.data3)
        bmrk.predict(data=self.dataT3)
        np.testing.assert_equal(model.ytH, model.yt)
        np.testing.assert_equal(bmrk.ytH, bmrk.yt)
        np.testing.assert_equal(model.model.best_params_, bmrk.model.best_params_)
        self.assertEqual(model.model.best_params_['clf__n_neighbors'], 1)

if __name__ == '__main__':
    unittest.main()
