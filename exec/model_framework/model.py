"""
Classification model class
"""

from abc import ABC, abstractmethod
from typing import Iterable, Dict, Callable, Tuple, Union
import numbers
import inspect

import os

os.environ['THEANO_FLAGS'] = "floatX=float32"

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
import pygam as gam
import pygam.utils as gamutils
from sklearn import base as skbase
from sklearn import ensemble as skens
from sklearn import feature_selection as skfs
from sklearn import linear_model as sklm
from sklearn import metrics as skmtcs
from sklearn import model_selection as skms
from sklearn import neighbors as sknbr
from sklearn import pipeline as skpipe
from sklearn import preprocessing as skprcss
from sklearn import tree as sktree
from sklearn import svm as sksvm
from sklearn.utils import estimator_checks as skutilcheck
from sklearn.utils import validation as skutilvalid
from statsmodels.nonparametric import kernel_regression as smkernel
from statsmodels.nonparametric import _kernel_base as smkernelbase
from theano import shared
import pymc3 as pm
from pymc3 import math as pmmath
# import xgboost

import exec.model_framework.utilmodel as utmdl


class Metrics(ABC):
    """
    Score method interface (compatible with Scikit-learn)
    """

    accuracy = (skmtcs.accuracy_score, False)
    accproba = (lambda y, yH: 1. - skmtcs.mean_absolute_error(y, yH[:, 1]), True)
    logproba = (lambda y, yH: -skmtcs.log_loss(y, yH[:, 1]), True)
    aucproba = (lambda y, yH: skmtcs.roc_auc_score(y, yH[:, 1]), True)
    precision = (skmtcs.precision_score, False)
    recall = (skmtcs.recall_score, False)

    @staticmethod
    def generator(method: str = 'accuracy') -> Tuple[Callable[[float, float], float], bool]:
        """
        Generate a score function from its name

        Args:
            method: score method name
        """
        if method == 'accuracy':
            return Metrics.accuracy
        elif method == 'accproba':
            return Metrics.accproba
        elif method == 'logproba':
            return Metrics.logproba
        elif method == 'aucproba':
            return Metrics.aucproba
        elif method == 'precision':
            return Metrics.precision
        elif method == 'recall':
            return Metrics.recall
        else:
            raise LookupError


class ModelAbs(ABC):
    """
    Abstract classification model

    Attributes:
        model:
        name:
        x:
        xt:
        y:
        yt:
        yH:
        ytH:
        yP:
        ytP:
    """

    def __init__(self, model, name: str):
        """
        Initialize a classifier

        Args:
            model: Base classifier
            name: Name of the classifier
        """

        self.model = model
        self.name = name

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the model to the data (features + labels)

        Args:
            data: DataFrame with **x** and **y**
        """

        self.x, self.y = utmdl.dataframeToXy(data)
        self.yH = None
        self.yP = None

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> None:
        """
        Predict labels for the chosen features

        Args:
            data: DataFrame with **xt** (and **yt**, optionally)
        """

        self.xt, self.yt = utmdl.dataframeToXy(data)
        self.ytH = None
        self.ytP = None

    def scoreIS(self, methods: Iterable[str] = ('accuracy',)) -> Dict[str, float]:
        """Score the classifier (in-sample)"""
        scores = {}
        for method in methods:
            metrics, proba = Metrics.generator(method=method)
            if proba:
                yH = self.yP
            else:
                yH = self.yH
            scores.update({method: metrics(self.y, yH)})
        return scores

    def scoreOOS(self, methods: Iterable[str] = ('accuracy',)) -> Dict[str, float]:
        """Score the classifier (out-of-sample)"""
        scores = {}
        for method in methods:
            metrics, proba = Metrics.generator(method=method)
            if proba:
                ytH = self.ytP
            else:
                ytH = self.ytH
            scores.update({method: metrics(self.yt, ytH) if (self.yt is not None) else None})
        return scores

    @abstractmethod
    def scoreCV(self, methods: Iterable[str] = ('accuracy',), cv: Union[None, int] = 5) -> Dict[str, float]:
        """Score the classifier (cross-validation)"""
        pass

    def confusionMatrix(self) -> (np.ndarray, np.ndarray):
        """Compute a confusion matrix of the classifier (IS and OOS)"""
        confIS = skmtcs.confusion_matrix(self.y, self.yH)
        confIS = confIS / confIS.sum().sum()
        if self.yt is not None:
            confOOS = skmtcs.confusion_matrix(self.yt, self.ytH)
            confOOS = confOOS / confOOS.sum().sum()
        else:
            confOOS = None
        return confIS, confOOS

    def printSetsInfo(self) -> None:
        """Print a summary on the training and testing data sets"""
        sampleSize = len(self.y)
        sampleSizeT = len(self.yt) if (self.yt is not None) else None
        posRate = np.sum(self.y) / len(self.y)
        posRateT = np.sum(self.yt) / len(self.yt) if (self.yt is not None) else None
        print('-----Train and Test Sets-----')
        print('Sample Size (Train / Test): {:d} / {:d}'.format(sampleSize, sampleSizeT))
        print('Survived Rate (Train / Test): {:.2f} / {:.2f}'.format(posRate, posRateT))
        print('')

    def printConfusion(self) -> None:
        """Print the confusion matrices"""
        confIS, confOOS = self.confusionMatrix()
        confISdf = pd.DataFrame(confIS, index=['0', '1'], columns=['0-pred', '1-pred'])
        confOOSdf = pd.DataFrame(confOOS, index=['0', '1'], columns=['0-pred', '1-pred'])
        pd.options.display.float_format = '{:.2f}'.format
        print('-----Confusion (IS)-----')
        print(confISdf)
        print('-----Confusion (OOS)-----')
        print(confOOSdf)
        print('')

    def printPerformance(self, cv: Union[None, int] = 5):
        methods = ('accuracy', 'accproba', 'logproba', 'aucproba', 'recall', 'precision')
        scoreIS = self.scoreIS(methods)
        scoreOOS = self.scoreOOS(methods)
        scoreCV = self.scoreCV(methods, cv=cv) if cv is not None else {x: np.nan for x in methods}
        print('-----Performance-----')
        for method in methods:
            print('{}\t (IS / CV / OOS): {:.2f} / {:.2f} / {:.2f}'.format(method, scoreIS[method],
                                                                          scoreCV[method], scoreOOS[method]))
        print('')

    @abstractmethod
    def printSummary(self) -> None:
        """Print a summary on the classifier"""
        pass

    def plotROC(self) -> None:
        """Plot the ROC curve"""
        fig, ax = plt.subplots()
        self.staticPlotROC(self.y, self.yP[:, 1], ax=ax, label='IS', title='ROC: {}'.format(self.name))
        self.staticPlotROC(self.yt, self.ytP[:, 1], ax=ax, label='OOS', title='ROC: {}'.format(self.name))
        fig.set_tight_layout(True)
        fig.show()

    @staticmethod
    def staticPlotROC(y: pd.DataFrame, yP: pd.DataFrame, ax=None, label: str = ' ', title: str = 'ROC') -> None:
        """Plot the ROC curve (static)"""

        def plotAux():
            fpr, tpr, _ = skmtcs.roc_curve(y, yP)
            ax.plot(fpr, tpr, label=label)
            ax.set_title(title)
            ax.set_xlabel('False Positive')
            ax.set_ylabel('True Positive')
            ax.legend()

        if ax is None:
            fig, ax2 = plt.subplots()
            plotAux()
            fig.show()
        else:
            plotAux()

    # @staticmethod
    # def staticPredictionsCV(modelGen, modelKwargs, data, nsplits=5):
    #     kfold = skms.KFold(n_splits=nsplits, random_state=1)
    #     predictions = []
    #     for trainIdx, testIdx in kfold.split(data):
    #         modelTemp = modelGen(**modelKwargs)
    #         modelTemp.fit(data.iloc[trainIdx, :])
    #         modelTemp.predict(data.iloc[testIdx, :])
    #         predictions.append({'yt': modelTemp.yt,
    #                             'ytH': modelTemp.ytH,
    #                             'ytP': modelTemp.ytP})
    #     return predictions
    #
    # @staticmethod
    # def staticScoreCV(modelGen, modelKwargs, data, nsplits=5, method='accuracy', proba=False):
    #     predictions = ModelAbs.staticPredictionsCV(modelGen=modelGen, modelKwargs=modelKwargs, data=data,
    #                                                nsplits=nsplits)
    #     scoresCV = np.zeros(nsplits, dtype=np.float)
    #     for i, p in enumerate(predictions):
    #         y = p['yt']
    #         yH = p['ytP'] if proba else p['ytH']
    #         scoresCV[i] = ModelAbs.staticScore(y=y, yH=yH, method=method)
    #     return scoresCV

    @staticmethod
    def balancedWeights(y):
        posRate = np.sum(y) / len(y)
        weights = y * (1. - posRate) + (1. - y) * posRate
        return weights

    def _getFeatureNames(self):
        """Return names of the original features"""
        return self.x.columns.values


class ModelNormalAbs(ModelAbs):
    """
    Abstract sklearn classifier (the base classifier is a pipeline, with predict_proba available)
    """

    def __init__(self, model: skpipe.Pipeline, name: str):
        ModelAbs.__init__(self, model, name)

    def fit(self, data):
        self.x, self.y = utmdl.dataframeToXy(data)
        self.model.fit(X=self.x, y=self.y)
        self.yH = self.model.predict(X=self.x)
        self.yP = self.model.predict_proba(X=self.x)

    def predict(self, data):
        self.xt, self.yt = utmdl.dataframeToXy(data)
        self.ytH = self.model.predict(X=self.xt)
        self.ytP = self.model.predict_proba(X=self.xt)

    def scoreCV(self, methods: Iterable[str] = ('accuracy',), cv: int = 5, random_state: int = 1) -> Dict[str, float]:
        scoring = {}
        for method in methods:
            metrics, proba = Metrics.generator(method=method)
            cvScorer = skmtcs.make_scorer(metrics, needs_proba=proba)
            scoring.update({method: cvScorer})

        kfold = skms.KFold(n_splits=cv, random_state=random_state)
        scoresRaw = skms.cross_validate(estimator=self.model, X=self.x.values, y=self.y, cv=kfold,
                                        scoring=scoring, return_train_score=False)
        scores = {}
        for method in methods:
            scores.update({method: scoresRaw['test_' + method].mean()})
        return scores

    @abstractmethod
    def _getClassifier(self) -> skbase.ClassifierMixin:
        """Return Sklearn base classifier"""
        pass


class LogisticAbs(ModelNormalAbs):
    """
    Abstract Logistic classifier
    """

    def __init__(self, model, name):
        ModelNormalAbs.__init__(self, model=model, name=name)

    @abstractmethod
    def _getClassifier(self) -> sklm.LogisticRegression:
        pass

    def printCoefficients(self) -> None:
        """Print the fitted coefficients"""
        coef = self._getClassifier().coef_[0]
        assert len(coef) == len(self._getFeatureNames())
        coefSrs = pd.Series(coef, index=self._getFeatureNames())
        print('-----Coefficients-----')
        print(coefSrs)
        print('')


class Logistic(LogisticAbs):
    """
    Logistic classifier
    """

    def __init__(self, scale: bool = True, fit_intercept: bool = False, C: int = 1.):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = sklm.LogisticRegression(fit_intercept=fit_intercept, C=C, solver='lbfgs', penalty='l2')
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        LogisticAbs.__init__(self, model=model, name='Logistic')

    def _getClassifier(self):
        return self.model.named_steps['clf']

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printCoefficients()
        self.printPerformance()
        self.printConfusion()


class LogisticRidgeCV(LogisticAbs):
    """
    Logistic classifier with Ridge CV penalty
    Todo: Logistic Ridge with a specified number of degrees of freedom
    """

    def __init__(self, scale: bool = True, fit_intercept: bool = False, Cs: int = 10):
        self.name = 'ABC Logistic Regression CV'
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = sklm.LogisticRegressionCV(fit_intercept=fit_intercept, Cs=Cs, cv=10, solver='lbfgs',
                                               penalty='l2')
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        LogisticAbs.__init__(self, model=model, name='Logistic Ridge')

    def _getClassifier(self):
        return self.model.named_steps['clf']

    def printRidgeMultiplier(self) -> None:
        print('-----Ridge CV multiplier-----')
        print('Ridge Multiplier = {:.2f}'.format(self.model.named_steps['clf'].C_[0]))
        print('')

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printCoefficients()
        self.printRidgeMultiplier()
        self.printPerformance()
        self.printConfusion()


class LogisticBestSubset(LogisticAbs):
    """
    Logistic classifier with k features pre-selected
    """

    def __init__(self, scale: bool = True, fit_intercept: bool = False, k: int = 5, C: int = 1.):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        featureSelector = skfs.SelectKBest(score_func=skfs.f_classif, k=k)
        classifier = sklm.LogisticRegression(fit_intercept=fit_intercept, C=C, solver='lbfgs', penalty='l2')
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('fselect', featureSelector), ('clf', classifier)])
        LogisticAbs.__init__(self, model=model, name='Logistic kBest')

    def _getClassifier(self):
        return self.model.named_steps['clf']

    def printCoefficients(self):
        coef = self._getClassifier().coef_[0]
        support = self.model.named_steps['fselect'].get_support()
        assert len(coef) == len(self._getFeatureNames()[support])
        coefSrs = pd.Series(coef, index=self._getFeatureNames()[support])
        print('-----Coefficients-----')
        print(coefSrs)
        print('')

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printCoefficients()
        self.printPerformance()
        self.printConfusion()


class _LogisticGAMSklearn(gam.LogisticGAM):
    """
    Sklearn-compatible Additive Logistic base classifier
    """

    def __init__(self, lam=0.6, max_iter=100, n_splines=25, spline_order=3,
                 penalties='auto', dtype='auto', tol=1e-4,
                 callbacks=('deviance', 'diffs', 'accuracy'),
                 fit_intercept=True, fit_linear=False, fit_splines=True,
                 constraints=None):
        gam.LogisticGAM.__init__(self, lam=lam, max_iter=max_iter, n_splines=n_splines, spline_order=spline_order,
                                 penalties=penalties, dtype=dtype, tol=tol,
                                 callbacks=callbacks,
                                 fit_intercept=fit_intercept, fit_linear=fit_linear, fit_splines=fit_splines,
                                 constraints=constraints)

    def get_params(self, deep=False):
        params = gam.LogisticGAM.get_params(self, deep=deep)
        del params['verbose']
        return params

    def predict_proba(self, X):
        proba = gam.LogisticGAM.predict_proba(self, X)
        skProba = np.zeros((len(proba), 2), dtype=float)
        skProba[:, 1] = proba
        skProba[:, 0] = 1 - proba
        return skProba


class LogisticGAM(ModelNormalAbs):
    """
    Additive Logistic classifier
    """

    def __init__(self, scale=True, fit_intercept=False, n_splines=15, lam=1., constraints=None):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = _LogisticGAMSklearn(fit_intercept=fit_intercept, n_splines=n_splines, lam=lam,
                                         constraints=constraints)
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        ModelNormalAbs.__init__(self, model=model, name='Logistic GAM')

    def _getClassifier(self) -> gam.LogisticGAM:
        return self.model.named_steps['clf']

    def printStatistics(self):
        print('-----Statistics-----')
        data = []
        for i in np.arange(len(self._getClassifier()._n_splines)):
            data.append({
                'feature_func': '{:_<15}'.format(str(self._getFeatureNames()[i])),
                'n_splines': self._getClassifier()._n_splines[i],
                'spline_order': self._getClassifier()._spline_order[i],
                'fit_linear': self._getClassifier()._fit_linear[i],
                'dtype': self._getClassifier()._dtype[i],
                'lam': np.round(self._getClassifier()._lam[i + self._getClassifier().fit_intercept], 4),
                'p_value': '%.2e' % (
                    self._getClassifier().statistics_['p_values'][i + self._getClassifier().fit_intercept]),
                'sig_code': gamutils.sig_code(
                    self._getClassifier().statistics_['p_values'][i + self._getClassifier().fit_intercept])
            })
        if self._getClassifier().fit_intercept:
            data.append({
                'feature_func': 'intercept',
                'n_splines': '',
                'spline_order': '',
                'fit_linear': '',
                'dtype': '',
                'lam': '',
                'p_value': '%.2e' % (self._getClassifier().statistics_['p_values'][0]),
                'sig_code': gamutils.sig_code(self._getClassifier().statistics_['p_values'][0])
            })
        fmt = [
            ('Feature Function', 'feature_func', 18),
            ('Data Type', 'dtype', 14),
            ('Num Splines', 'n_splines', 13),
            ('Spline Order', 'spline_order', 13),
            ('Linear Fit', 'fit_linear', 11),
            ('Lambda', 'lam', 10),
            ('P > x', 'p_value', 10),
            ('Sig. Code', 'sig_code', 10)
        ]
        print(gamutils.TablePrinter(fmt, ul='=')(data))
        print("=" * 106)
        print("Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        print()

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printStatistics()
        self.printPerformance()
        self.printConfusion()

    def plotFeatureFit(self) -> None:
        """
        Partial dependence plot (?)
        Todo: Generic Partial Dependence plot and compare
        """
        gridGAM = gamutils.generate_X_grid(self._getClassifier())
        # plt.rcParams['figure.figsize'] = (28, 8)
        fig, axs = plt.subplots(1, len(self._getFeatureNames()))
        titles = self._getFeatureNames()
        for i, ax in enumerate(axs):
            pdep, confi = self._getClassifier().partial_dependence(gridGAM, feature=i, width=.9)
            ax.plot(gridGAM[:, i], pdep)
            ax.plot(gridGAM[:, i], confi[0][:, 0], c='grey', ls='--')
            ax.plot(gridGAM[:, i], confi[0][:, 1], c='grey', ls='--')
            ax.set_title(titles[i])
        fig.set_tight_layout(True)
        fig.show()


class _LogisticLinearLocalSklearn(skbase.BaseEstimator, skbase.ClassifierMixin):
    """
    Sklearn-compatible Local Logistic classifier (using Local Linear as a proxy)
    """

    def __init__(self, reg_type: str = 'll', bw: Union[str, float, Iterable] = 'cv_ls'):
        self.reg_type = reg_type
        self.bw = bw

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray, Iterable]) \
            -> skbase.ClassifierMixin:
        X, y = self._check_X_y_fit(X, y)
        # self.classes_ = skutilmult.unique_labels(y)
        self.nfeatures_ = X.shape[1]
        bw = np.full(self.nfeatures_, self.bw) if isinstance(self.bw, numbers.Number) else self.bw

        self.model_ = smkernel.KernelReg(endog=y * 2 - 1, exog=X, var_type='c' * self.nfeatures_,
                                         reg_type=self.reg_type, bw=bw,
                                         defaults=smkernelbase.EstimatorSettings(efficient=False))
        return self

    def decision_function(self, X) -> np.ndarray:
        skutilvalid.check_is_fitted(self, ['model_'])
        X = self._check_X_predict(X)
        dsn_pred, mgn_pred = self.model_.fit(data_predict=X)
        return dsn_pred

    def predict(self, X) -> np.ndarray:
        skutilvalid.check_is_fitted(self, ['model_'])
        dsn_pred = self.decision_function(X)
        y_pred = (dsn_pred > 0).astype(int)
        return y_pred

    def predict_proba(self, X) -> np.ndarray:
        skutilvalid.check_is_fitted(self, ['model_'])
        dsn_pred = self.decision_function(X)
        proba_pred = np.zeros((X.shape[0], 2), dtype=np.float)
        proba_pred[:, 1] = 1 / (1 + np.exp(-dsn_pred))
        proba_pred[:, 0] = 1 - proba_pred[:, 0]
        return proba_pred

    def _check_X_y_fit(self, X, y):
        X, y = skutilvalid.check_X_y(X, y)
        assert np.all(np.unique(y) == np.array([0, 1]))
        return X, y

    def _check_X_predict(self, X):
        X = skutilvalid.check_array(X)
        assert X.shape[1] == self.nfeatures_, "Wrong X shape"
        return X

    # def score(self, X, y, sample_weight=None):
    #     pass


class LogisticLinearLocal(ModelNormalAbs):
    """
    Local Logistic classifier (Linear Proxy)
    Todo: Genuine Local Logistic with weights and add statistics
    """

    def __init__(self, scale=True, reg_type: str = 'll', bw: Union[str, float, Iterable] = 'cv_ls'):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = _LogisticLinearLocalSklearn(reg_type=reg_type, bw=bw)
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        ModelNormalAbs.__init__(self, model=model, name='Logistic Local (Linear Proxy)')

    def _getClassifier(self) -> _LogisticLinearLocalSklearn:
        return self.model.named_steps['clf']

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printPerformance()
        self.printConfusion()


class _LogisticBayesianSklearn(skbase.BaseEstimator, skbase.ClassifierMixin):
    """
    Sklearn-compatible Bayesian Logistic classifier
    """

    def __init__(self, featuresSd=10, nsamplesFit=200, nsamplesPredict=100, mcmc=True,
                 nsampleTune=200, discardTuned=True, samplerStep=None, samplerInit='auto'):
        self.featuresSd = featuresSd
        self.nsamplesFit = nsamplesFit
        self.nsamplesPredict = nsamplesPredict
        self.mcmc = mcmc
        self.nsampleTune = nsampleTune
        self.discardTuned = discardTuned
        self.samplerStep = samplerStep
        self.samplerInit = samplerInit

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray, Iterable]) \
            -> skbase.ClassifierMixin:
        import copy
        X, y = self._check_X_y_fit(X, y)
        self.X_shared_ = shared(X)
        self.y_shared_ = shared(y)
        self.nfeatures_ = X.shape[1]
        self.model_ = pm.Model(name='')
        # self.model_.Var('beta', pm.Normal(mu=0, sd=self.featuresSd))
        with self.model_:
            beta = pm.Normal('beta', mu=0, sd=self.featuresSd, testval=0, shape=self.nfeatures_)
            # mu = pm.Deterministic('mu', var=pmmath.dot(beta, self.X_shared_.T))
            mu = pmmath.dot(beta, self.X_shared_.T)
            y_obs = pm.Bernoulli('y_obs', p=pm.invlogit(mu), observed=self.y_shared_)
            if self.mcmc:
                self.trace_ = pm.sample(draws=self.nsamplesFit, tune=self.nsampleTune,
                                        discard_tuned_samples=self.discardTuned,
                                        step=self.samplerStep, init=self.samplerInit, progressbar=True)
            else:
                approx = pm.fit(method='advi')
                self.trace_ = approx.sample(draws=self.nsamplesFit)
        return self

    def decision_function(self, X) -> np.ndarray:
        skutilvalid.check_is_fitted(self, ['model_'])
        X = self._check_X_predict(X)
        self.X_shared_.set_value(X)
        self.y_shared_.set_value(np.zeros(X.shape[0], dtype=np.int))
        with self.model_:
            post_pred = pm.sample_ppc(trace=self.trace_, samples=self.nsamplesPredict,
                                      progressbar=False)['y_obs'].mean(axis=0)
        return post_pred

    def predict(self, X) -> np.ndarray:
        skutilvalid.check_is_fitted(self, ['model_'])
        dsn_pred = self.decision_function(X)
        y_pred = (dsn_pred > 0.5).astype(int)
        return y_pred

    def predict_proba(self, X) -> np.ndarray:
        skutilvalid.check_is_fitted(self, ['model_'])
        dsn_pred = self.decision_function(X)
        proba_pred = np.zeros((X.shape[0], 2), dtype=np.float)
        proba_pred[:, 1] = dsn_pred
        proba_pred[:, 0] = 1 - proba_pred[:, 0]
        return proba_pred

    def _check_X_y_fit(self, X, y):
        X, y = skutilvalid.check_X_y(X, y)
        assert np.all(np.unique(y) == np.array([0, 1]))
        return X, y

    def _check_X_predict(self, X):
        X = skutilvalid.check_array(X)
        assert X.shape[1] == self.nfeatures_, "Wrong X shape"
        return X


class LogisticBayesian(ModelNormalAbs):
    """
    Bayesian Logistic
    Todo: Plot feature names in Bayesian LR
    """

    def __init__(self, scale=True, featuresSd=10, nsamplesFit=200, nsamplesPredict=100, mcmc=True,
                 nsampleTune=200, discardTuned=True, samplerStep=None, samplerInit='auto'):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = _LogisticBayesianSklearn(featuresSd=featuresSd, nsamplesFit=nsamplesFit,
                                              nsamplesPredict=nsamplesPredict, mcmc=mcmc,
                                              nsampleTune=nsampleTune, discardTuned=discardTuned,
                                              samplerStep=samplerStep, samplerInit=samplerInit)
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        ModelNormalAbs.__init__(self, model=model, name='Logistic Bayesian')

    def _getClassifier(self) -> _LogisticBayesianSklearn:
        return self.model.named_steps['clf']

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printPerformance(cv=5)
        self.printConfusion()

    def plotTrace(self):
        pm.traceplot(self._getClassifier().trace_)

    def plotPosterior(self):
        pm.plot_posterior(self._getClassifier().trace_)


class KNN(ModelNormalAbs):
    """
    kNN classifier
    """

    def __init__(self, scale: bool = True, n_neighbors: int = 10, weights: str = 'uniform'):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = sknbr.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric='minkowski')
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        ModelNormalAbs.__init__(self, model=model, name='kNN')

    def _getClassifier(self) -> sknbr.KNeighborsClassifier:
        return self.model.named_steps['clf']

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printPerformance()
        self.printConfusion()


class KNNCV(KNN):
    """
    kNN classifier with CV for k
    Todo: Check that CV does not give an optimistic score when random states are chained
    """

    def __init__(self, scale: bool = True, weights: str = 'uniform'):
        grid = {'clf__n_neighbors': (5, 10, 20, 40)}
        KNN.__init__(self, scale=scale, n_neighbors=grid['clf__n_neighbors'][0], weights=weights)
        self.model = skms.GridSearchCV(self.model, param_grid=grid, scoring='accuracy', cv=5)
        self.name = 'kNN CV'

    def _getClassifier(self):
        return self.model.best_estimator_.named_steps['clf']

    def printBestK(self) -> None:
        print('-----Best k-----')
        print('k = {:d} with the score = {:.2f}'.format(self.model.best_params_['clf__n_neighbors'],
                                                        self.model.best_score_))
        print('')

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printBestK()
        self.printPerformance()
        self.printConfusion()


class Tree(ModelNormalAbs):
    """
    Decision Tree classifier (CART algorithm)
    """

    def __init__(self, scale: bool = True, max_depth: Union[int, None] = 3, max_leaf_nodes: Union[int, None] = None,
                 class_weight: Union[None, str] = 'balanced', random_state: Union[int, None] = 1):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = sktree.DecisionTreeClassifier(criterion='gini', max_depth=max_depth, max_leaf_nodes=max_leaf_nodes,
                                                   class_weight=class_weight,
                                                   splitter='best', min_samples_leaf=5, random_state=random_state)
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        ModelNormalAbs.__init__(self, model=model, name='Decision Tree')

    def _getClassifier(self) -> sktree.DecisionTreeClassifier:
        return self.model.named_steps['clf']

    def printFeatureImportance(self):
        print('-----Feature Importance-----')
        importance = self._getClassifier().feature_importances_
        importanceSrs = pd.Series(importance, index=self.x.columns)
        print(importanceSrs)
        print('')

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printFeatureImportance()
        self.printPerformance()
        self.printConfusion()

    def visualizeTree(self) -> None:
        dotData = sktree.export_graphviz(self._getClassifier(), precision=2, proportion=True,
                                         feature_names=self._getFeatureNames(), class_names=['Dead', 'Alive'],
                                         impurity=True, filled=True, out_file=None)
        imgPath = os.path.join('data', 'temp', 'tree.png')
        pydotplus.graph_from_dot_data(dotData).write_png(imgPath)
        image = img.imread(imgPath)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title('Graph: {}'.format(self.name))
        fig.set_tight_layout(True)
        fig.show()


class TreeCV(Tree):
    """
    Decision Tree classifier with CV for max depth and max number of leaves
    """

    def __init__(self, scale: bool = True, class_weight: Union[None, str] = 'balanced',
                 random_state: Union[int, None] = 1):
        grid = {'clf__max_depth': (2, 3, 4), 'clf__max_leaf_nodes': (4, 6, 8, 12)}
        Tree.__init__(self, scale=scale, max_depth=grid['clf__max_depth'][0],
                      class_weight=class_weight, random_state=random_state)
        self.model = skms.GridSearchCV(self.model, param_grid=grid, scoring='accuracy', cv=5)
        self.name = 'Tree CV'

    def printParameters(self):
        print('-----Best Parameters-----')
        print('max_depth = {:d} and max_leaf_nodes = {:d} with the score = {:.2f}'.
              format(self.model.best_params_['clf__max_depth'] + 1, self.model.best_params_['clf__max_leaf_nodes'],
                     self.model.best_score_))
        print('')

    def _getClassifier(self):
        return self.model.best_estimator_.named_steps['clf']

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printParameters()
        self.printFeatureImportance()
        self.printPerformance()
        self.printConfusion()


class RandomForest(ModelNormalAbs):
    """
    Random Forest classifier (Decision Trees + Bagging)
    """

    def __init__(self, scale: bool = True, n_estimators: int = 128, max_features: Union[int, None] = None,
                 max_depth: Union[int, None] = None, max_leaf_nodes: Union[int, None] = 12,
                 class_weight: Union[None, str] = 'balanced', random_state: Union[int, None] = 1):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = skens.RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                                  max_leaf_nodes=max_leaf_nodes,
                                                  bootstrap=True, criterion='gini', max_depth=max_depth,
                                                  class_weight=class_weight,
                                                  min_samples_leaf=5, random_state=random_state)
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        ModelNormalAbs.__init__(self, model=model, name='Random Forest')

    def _getClassifier(self) -> skens.RandomForestClassifier:
        return self.model.named_steps['clf']

    def printFeatureImportance(self):
        print('-----Feature Importance-----')
        importance = self._getClassifier().feature_importances_
        importanceSrs = pd.Series(importance, index=self._getFeatureNames())
        print(importanceSrs)
        print('')

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printFeatureImportance()
        self.printPerformance()
        self.printConfusion()


class BoostedTree(ModelNormalAbs):
    """
    Boosted Trees (Gradient Boosting)
    """

    def __init__(self, scale: bool = True, n_estimators: int = 128, loss: str = 'deviance', learning_rate: float = 1.,
                 subsample: float = 1., max_features: Union[int, None] = None,
                 max_depth: Union[int, None] = 2, max_leaf_nodes: Union[int, None] = None,
                 random_state: int = 1, balanceWeights: bool = False):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        if not balanceWeights:
            classifier = skens.GradientBoostingClassifier(n_estimators=n_estimators, loss=loss,
                                                          learning_rate=learning_rate, subsample=subsample,
                                                          max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                                          max_depth=max_depth, min_samples_leaf=5,
                                                          random_state=random_state)
        else:
            class GradientBoostingClassifierBalanced(skens.GradientBoostingClassifier):
                def fit(self, X, y, sample_weight=None, monitor=None):
                    weights = ModelAbs.balancedWeights(y)
                    return skens.GradientBoostingClassifier.fit(self, X=X, y=y, sample_weight=weights)

            classifier = GradientBoostingClassifierBalanced(n_estimators=n_estimators, loss=loss,
                                                            learning_rate=learning_rate, subsample=subsample,
                                                            max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                                            max_depth=max_depth, min_samples_leaf=5,
                                                            random_state=random_state)

        # classifier = skens.GradientBoostingClassifier(n_estimators=n_estimators, loss=loss,
        #                                               learning_rate=learning_rate, subsample=subsample,
        #                                               max_features=max_features, max_leaf_nodes=max_leaf_nodes,
        #                                               max_depth=max_depth, min_samples_leaf=5,
        #                                               random_state=random_state)
        # if balanceWeights:
        #     def fitBalanced(self, X, y, sample_weight=None, monitor=None):
        #         weights = ModelAbs.balancedWeights(y)
        #         return self.fit(X=X, y=y, sample_weight=weights, monitor=monitor)
        #
        #     classifier.fit = MethodType(fitBalanced, classifier)

        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        ModelNormalAbs.__init__(self, model=model, name='Boosted Tree')

    def _getClassifier(self) -> skens.GradientBoostingClassifier:
        return self.model.named_steps['clf']

    def printFeatureImportance(self):
        print('-----Feature Importance-----')
        importance = self._getClassifier().feature_importances_
        importanceSrs = pd.Series(importance, index=self._getFeatureNames())
        print(importanceSrs)
        print('')

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printFeatureImportance()
        self.printPerformance()
        self.printConfusion()

    def plotPartialDependence(self, features=None) -> None:
        """Partial Dependence Plot"""
        if features is None: features = self._getFeatureNames()
        xScaled = self.model.named_steps['scaler'].transform(self.x)
        fig, axs = skens.partial_dependence.plot_partial_dependence(self._getClassifier(), X=xScaled, features=features,
                                                                    feature_names=self._getFeatureNames(),
                                                                    percentiles=(0.05, 0.95), grid_resolution=100)
        # fig.suptitle('Partial Dependence: {}'.format(self.name))
        # fig.subplots_adjust(top=0.9)
        fig.set_tight_layout(True)
        fig.show()


class BoostedTreeXGBoost(ModelNormalAbs):
    """
    Boosted Trees (XGBoost)
    UNDER DEVELOPMENT
    Todo: Complete BoostedTreeXGBoost (fix import xgboost and add balanced weight scaling)
    """
    pass


#     def __init__(self, scale: bool = True, n_estimators: int = 128, loss: str = 'deviance', learning_rate: float = 1.,
#                  subsample: int = 1., max_features: Union[int, None] = None,
#                  max_depth: Union[int, None] = 2, max_leaf_nodes: Union[int, None] = None,
#                  random_state: int = 1, balanceWeights: bool = False):
#         scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
#
#         if balanceWeights:
#             scale_pos_weight = 1
#         else:
#             scale_pos_weight = 1
#         classifier = xgboost.XGBClassifier(n_estimators=n_estimators, loss=loss,
#                                            learning_rate=learning_rate, subsample=subsample,
#                                            max_features=max_features, max_leaf_nodes=max_leaf_nodes,
#                                            max_depth=max_depth, min_samples_leaf=5, scale_pos_weight=scale_pos_weight,
#                                            random_state=random_state)
#
#         model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
#         ModelNormalAbs.__init__(self, model=model, name='Boosted Tree (XGBoost)')
#
#     def _getClassifier(self) -> xgboost.XGBClassifier:
#         return self.model.named_steps['clf']
#
#     def printFeatureImportance(self):
#         print('-----Feature Importance-----')
#         importance = self._getClassifier().feature_importances_
#         importanceSrs = pd.Series(importance, index=self._getFeatureNames())
#         print(importanceSrs)
#         print('')
#
#     def printSummary(self):
#         print('****** {} ******\n'.format(str.upper(self.name)))
#         self.printSetsInfo()
#         self.printFeatureImportance()
#         self.printPerformance()
#         self.printConfusion()


class _SVMSklearn(sksvm.SVC):
    """
    Base Sklearn SVM classifer with a faster (but very approximate) predict_proba function
    """
    def predict_proba(self, X) -> np.ndarray:
        dsn_pred = self.decision_function(X)
        proba_pred = np.zeros((X.shape[0], 2), dtype=np.float)
        proba_pred[:, 1] = 1 / (1 + np.exp(-dsn_pred))
        proba_pred[:, 0] = 1 - proba_pred[:, 0]
        return proba_pred


class SVM(ModelNormalAbs):
    """
    SVM classifier
    """

    def __init__(self, scale: bool = True, C: float = 1., kernel: str = 'poly', degree: int = 2,
                 gamma: Union[str, float] = 'auto',
                 class_weight: Union[None, str] = 'balanced', random_state: Union[int, None] = 1):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = _SVMSklearn(C=C, kernel=kernel, degree=degree, gamma=gamma, class_weight=class_weight,
                                 random_state=random_state, probability=False)
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        ModelNormalAbs.__init__(self, model=model, name='SVM ({}):'.format(kernel))

    def _getClassifier(self) -> sksvm.SVC:
        return self.model.named_steps['clf']

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printPerformance()
        self.printConfusion()


class SVMCV(SVM):
    """
    SVM classifier with CV for C
    Todo: Add other SVM parameters (gamma, degree) to the grid search
    Todo: Add a CV-classifier generator
    """

    def __init__(self, scale: bool = True, kernel: str = 'poly', degree: int = 2, gamma: Union[str, float] = 'auto',
                 class_weight: Union[None, str] = 'balanced', random_state: Union[int, None] = 1):
        grid = {'clf__C': (0.01, 0.1, 1, 10, 100)}
        SVM.__init__(self, scale=scale, C=grid['clf__C'][0], kernel=kernel, degree=degree, gamma=gamma,
                     class_weight=class_weight, random_state=random_state)
        self.model = skms.GridSearchCV(self.model, param_grid=grid, scoring='accuracy', cv=5)
        self.name = 'SVM ({}):'.format(kernel)

    def _getClassifier(self) -> sksvm.SVC:
        return self.model.best_estimator_.named_steps['clf']

    def printBestC(self) -> None:
        print('-----Best C-----')
        print('C = {:.2f} with the score = {:.2f}'.format(self.model.best_params_['clf__C'], self.model.best_score_))
        print('')

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printBestC()
        self.printPerformance()
        self.printConfusion()


if __name__ == '__main__':
    import mkl
    # import pygpu
    import theano
    import sklearn

    print('Sklearn v. {}'.format(sklearn.__version__))
    print('PyMC3 v. {}'.format(pm.__version__))
    print('Theano v. {}'.format(theano.__version__))
    print('...device = {}'.format(theano.config.device))
    print('...floatX = {}'.format(theano.config.floatX))
    print('...blas.ldflags = {}'.format(theano.config.blas.ldflags))
    print('...blas.check_openmp = {}'.format(theano.config.blas.check_openmp))
    # print('pygpy v. {}'.format(pygpu.__version__))
    # print('MKL v. {}'.format(mkl.__version__))
    # theano.test()
    print(os.getcwd())
    model = pm.Model(name='')
    # self.model_.Var('beta', pm.Normal(mu=0, sd=self.featuresSd))
    with model:
        beta = pm.Normal(name='beta', mu=0, sd=4)
    print(model)
