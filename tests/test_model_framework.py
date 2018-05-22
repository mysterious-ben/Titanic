import unittest
import modules.model_framework.sklearn_model as skmdl
import modules.model_framework.model as mdl

import numpy as np
import pandas as pd
from sklearn import preprocessing as skprcss


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

    def test__Scaler_array(self):
        features = [0, 1]
        featuresSup = list(set(range(3)) - set(features))
        for v1, d1, d2 in [[1, self.data1.values, self.dataT1.values],
                                [2, self.data2.values, self.dataT2.values]]:
            scaler = skmdl._Scaler(copy=True, with_mean=True, with_std=True, features=features)
            scalerBmrk = skprcss.StandardScaler(copy=True, with_mean=True, with_std=True)
            scaler.fit(X=d1)
            scalerBmrk.fit(X=d1[:, features])
            for v2, dtest in enumerate([d1, d2]):
                X = scaler.transform(X=dtest)
                XBmrk = scalerBmrk.transform(X=dtest[:, features])
                with self.subTest("{}-{}: scaled features".format(v1, v2)):
                    np.testing.assert_equal(X[:, features], XBmrk)
                with self.subTest("{}-{}: unscaled features".format(v1, v2)):
                    np.testing.assert_equal(X[:, featuresSup], dtest[:, featuresSup])

    def test__Scaler_df(self):
        features = [0, 1]
        featuresSup = list(set(range(3)) - set(features))
        for version, d1, d2 in [[1, self.data1, self.dataT1],
                                [2, self.data2, self.dataT2]]:
            scaler = skmdl._Scaler(copy=True, with_mean=True, with_std=True, features=features)
            scalerBmrk = skprcss.StandardScaler(copy=True, with_mean=True, with_std=True)
            scaler.fit(X=d1)
            scalerBmrk.fit(X=d1.iloc[:, features])
            X = scaler.transform(X=d2)
            XBmrk = scalerBmrk.transform(X=d2.iloc[:, features])
            with self.subTest("{}: scaled features".format(version)):
                np.testing.assert_equal(X[:, features], XBmrk)
            with self.subTest("{}: unscaled features".format(version)):
                np.testing.assert_equal(X[:, featuresSup], d2.iloc[:, featuresSup])

    def test_Logistic_prediction_1(self):
        for scale in ['none', 'all']:
            with self.subTest(scale=scale):
                model = mdl.Logistic(scale=scale, fit_intercept=False)
                model.fit(data=self.data1)
                model.predict(data=self.dataT1)
                self.assertGreater(model._getBaseClassifier().coef_[0][0], 0)
                self.assertAlmostEqual(model._getBaseClassifier().coef_[0][1], 0)
                np.testing.assert_equal(model.ytH, model.yt)

    def test_Logistic_prediction_2(self):
        for scale in ['none', 'all']:
            with self.subTest(scale=scale):
                model = mdl.Logistic(scale=scale, fit_intercept=False)
                model.fit(data=self.data2)
                model.predict(data=self.dataT2)
                self.assertGreater(model._getBaseClassifier().coef_[0][0], 0)
                self.assertLess(model._getBaseClassifier().coef_[0][1], 0)
                np.testing.assert_equal(model.ytH, model.yt)

    def test_KNN(self):
        for scale in ['none', 'all']:
            with self.subTest(scale=scale):
                model = mdl.KNN(scale=scale, n_neighbors=1)
                model.fit(data=self.data3)
                model.predict(data=self.dataT3)
                print(model._getScaler().scaler_.scale_, model._getScaler().scaler_.mean_)
                np.testing.assert_equal(model.yH, model.y)
                np.testing.assert_equal(model.ytH, model.yt)

    def test_genModelCV_benchmark_KNNCV(self):
        model = mdl.genModelCV(ModelClass=mdl.KNN, cv=2, grid={'n_neighbors': [1, 3]}) \
            (scale='all', weights='uniform')
        model.fit(data=self.data3)
        model.predict(data=self.dataT3)
        bmrk = mdl.KNNCV(cv=2, scale='all', weights='uniform', grid={'n_neighbors': np.arange(1, 3)})
        bmrk.fit(data=self.data3)
        bmrk.predict(data=self.dataT3)
        with self.subTest(description="best_params_: vs. benchmark"):
            np.testing.assert_equal(model.model.best_params_, bmrk.model.best_params_)
        with self.subTest(description="best_params_: vs. true"):
            self.assertEqual(model.model.best_params_['clf__n_neighbors'], 1)
        with self.subTest(description="yH: vs. benchmark"):
            np.testing.assert_equal(model.yH, bmrk.yH)
        with self.subTest(description="yH: vs. true"):
            np.testing.assert_equal(model.yH, model.y)
        with self.subTest(description="ytH: vs. benchmark"):
            np.testing.assert_equal(model.ytH, bmrk.ytH)
        with self.subTest(description="ytH: vs. true"):
            np.testing.assert_equal(model.ytH, model.yt)


if __name__ == '__main__':
    unittest.main()
