"""
Classification model class
"""

from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Dict, Callable, Union, Any, Type
import numbers
import inspect
import functools
import os
import copy

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
import pymc3 as pm
# import xgboost

import utils.utilgen as utgen
import exec.model_framework.utilmodel as utmdl
import exec.model_framework.sklearn_model as skmdl
import exec.model_framework.voting_classifier_cv as skvote


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

    def copy(self):
        return copy.deepcopy(self)

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

    def scoreIS(self, methods: Sequence[str] = ('accuracy',)) -> Dict[str, float]:
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

    def scoreOOS(self, methods: Sequence[str] = ('accuracy',)) -> Dict[str, float]:
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
    def scoreCV(self, methods: Sequence[str] = ('accuracy',), cv: Union[None, int] = 5) -> Dict[str, float]:
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
        """Print information on the training and testing data sets"""
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
        """Print information on classifier performance (IS / CV / OOS)"""
        if self.yP is None:
            print('<PROBABILITY IS NONE: SOME PERFORMANCE STATISTICS ARE NOT AVAILABLE>')
            methods = ('accuracy',)
        else:
            methods = ('accuracy', 'accproba', 'logproba', 'aucproba', 'recall', 'precision')
        scoreIS = self.scoreIS(methods)
        scoreOOS = self.scoreOOS(methods)
        scoreCV = self.scoreCV(methods, cv=cv) if cv is not None else {x: np.nan for x in methods}
        print('-----Performance-----')
        for method in methods:
            print('{}\t (IS / CV / OOS): {:.2f} / {:.2f} ({:.2f}) / {:.2f}'.
                  format(method, scoreIS[method], np.mean(scoreCV[method]), np.std(scoreCV[method]), scoreOOS[method]))
        print('')

    def printCoefficientsInfo(self):
        """Print information on classifier coefficients"""
        pass

    def printSummary(self, cv: Union[None, int] = 5) -> None:
        """Print a summary on the classifier"""
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printPerformance(cv=cv)
        self.printConfusion()
        self.printCoefficientsInfo()

    def plotROC(self) -> None:
        """Plot the ROC curve"""
        if self.yP is None:
            print('<PROBABILITY IS NONE: ROC PLOT IS NOT AVAILABLE>')
        else:
            fig, ax = plt.subplots()
            self.staticPlotROC(self.y, self.yP[:, 1], ax=ax, label='IS', title='ROC: {}'.format(self.name))
            self.staticPlotROC(self.yt, self.ytP[:, 1], ax=ax, label='OOS', title='ROC: {}'.format(self.name))
            fig.set_tight_layout(True)
            fig.show()

    def printPlotSummary(self, cv: Union[None, int] = 5) -> None:
        """Print a summary and show selected plots"""
        self.printSummary(cv=cv)
        self.plotROC()

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
        try:
            self.yP = self.model.predict_proba(X=self.x)
        except AttributeError:
            self.yP = None

    def predict(self, data):
        self.xt, self.yt = utmdl.dataframeToXy(data)
        self.ytH = self.model.predict(X=self.xt)
        try:
            self.ytP = self.model.predict_proba(X=self.xt)
        except AttributeError:
            self.ytP = None

    def scoreCV(self, methods: Sequence[str] = ('accuracy',), cv: int = 20, random_state: int = 1) \
            -> Dict[str, Sequence[float]]:
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
            scores.update({method: scoresRaw['test_' + method]})
        return scores

    @abstractmethod
    def _getBaseClassifier(self) -> skbase.ClassifierMixin:
        """Return Sklearn base classifier"""
        return self.model.named_steps['clf']

    def _getScaler(self) -> skprcss.StandardScaler:
        """Return Sklearn data scaler"""
        return self.model.named_steps['scaler']


def genModelCV(ModelClass: Type[ModelNormalAbs], grid: Dict[str, Any]):
    """
    Generate a classifier with some of the parameters selected by CV
    Todo: Imply doctring for ModelCV from ModelClass OR accept model instance instead of class

    Args:
        ModelClass: Classification model
        grid: Dictionary {parameter: value grid for CV}

    Returns:
        CV Classification model
    """

    class ModelCV(ModelClass):
        # noinspection PyMissingConstructor
        def __init__(self, cv: int, *args, **kwargs):
            self.grid = {'clf__' + x: grid[x] for x in grid}
            ModelClass.__init__(self, *args, **kwargs)
            self.model = skms.GridSearchCV(self.model, param_grid=self.grid, scoring='accuracy', cv=cv)
            self.name += ' CV'

        def _getBaseClassifier(self):
            return self.model.best_estimator_.named_steps['clf']

        def _getScaler(self) -> skprcss.StandardScaler:
            return self.model.best_estimator_.named_steps['scaler']

        def printBestParamCV(self) -> None:
            print('-----Best CV Parameters-----')
            for param in self.grid:
                print('{} = {:.2f}'.format(param[5:], self.model.best_params_[param]))
            scores = self.model.cv_results_['mean_test_score']
            print('...with the score = {:.2f}   | avg = {:.2f}, std = {:.2f}'
                  .format(np.max(scores), np.mean(scores), np.std(scores)))
            print('')

        def printCoefficientsInfo(self):
            ModelClass.printCoefficientsInfo(self)
            self.printBestParamCV()

    return ModelCV


class LogisticAbs(ModelNormalAbs):
    """
    Abstract Logistic classifier
    """

    def __init__(self, model, name):
        ModelNormalAbs.__init__(self, model=model, name=name)

    @abstractmethod
    def _getBaseClassifier(self) -> sklm.LogisticRegression:
        pass

    def printCoefficientsInfo(self) -> None:
        """Print the fitted coefficients"""
        coef = self._getBaseClassifier().coef_[0]
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

    def _getBaseClassifier(self) -> sklm.LogisticRegression:
        return self.model.named_steps['clf']


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

    def _getBaseClassifier(self) -> sklm.LogisticRegressionCV:
        return self.model.named_steps['clf']

    def printCoefficientsInfo(self) -> None:
        LogisticAbs.printCoefficientsInfo(self)
        print('-----Ridge CV multiplier-----')
        print('Ridge Multiplier = {:.2f}'.format(self.model.named_steps['clf'].C_[0]))
        print('')


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

    def _getBaseClassifier(self) -> sklm.LogisticRegression:
        return self.model.named_steps['clf']

    def printCoefficientsInfo(self):
        coef = self._getBaseClassifier().coef_[0]
        support = self.model.named_steps['fselect'].get_support()
        assert len(coef) == len(self._getFeatureNames()[support])
        coefSrs = pd.Series(coef, index=self._getFeatureNames()[support])
        print('-----Coefficients-----')
        print(coefSrs)
        print('')


class LogisticGAM(ModelNormalAbs):
    """
    Additive Logistic classifier
    """

    def __init__(self, scale=True, fit_intercept=False, n_splines=15, lam=1., constraints=None):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = skmdl._LogisticGAM(fit_intercept=fit_intercept, n_splines=n_splines, lam=lam,
                                        constraints=constraints)
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        ModelNormalAbs.__init__(self, model=model, name='Logistic GAM')

    def _getBaseClassifier(self) -> gam.LogisticGAM:
        return self.model.named_steps['clf']

    def printCoefficientsInfo(self):
        print('-----Statistics-----')
        data = []
        for i in np.arange(len(self._getBaseClassifier()._n_splines)):
            data.append({
                'feature_func': '{:_<15}'.format(str(self._getFeatureNames()[i])),
                'n_splines': self._getBaseClassifier()._n_splines[i],
                'spline_order': self._getBaseClassifier()._spline_order[i],
                'fit_linear': self._getBaseClassifier()._fit_linear[i],
                'dtype': self._getBaseClassifier()._dtype[i],
                'lam': np.round(self._getBaseClassifier()._lam[i + self._getBaseClassifier().fit_intercept], 4),
                'p_value': '%.2e' % (
                    self._getBaseClassifier().statistics_['p_values'][i + self._getBaseClassifier().fit_intercept]),
                'sig_code': gamutils.sig_code(
                    self._getBaseClassifier().statistics_['p_values'][i + self._getBaseClassifier().fit_intercept])
            })
        if self._getBaseClassifier().fit_intercept:
            data.append({
                'feature_func': 'intercept',
                'n_splines': '',
                'spline_order': '',
                'fit_linear': '',
                'dtype': '',
                'lam': '',
                'p_value': '%.2e' % (self._getBaseClassifier().statistics_['p_values'][0]),
                'sig_code': gamutils.sig_code(self._getBaseClassifier().statistics_['p_values'][0])
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

    def plotFeatureFit(self) -> None:
        """
        Partial dependence plot (?)
        Todo: Generic Partial Dependence plot and compare
        """
        gridGAM = gamutils.generate_X_grid(self._getBaseClassifier())
        # plt.rcParams['figure.figsize'] = (28, 8)
        fig, axs = plt.subplots(1, len(self._getFeatureNames()))
        titles = self._getFeatureNames()
        for i, ax in enumerate(axs):
            pdep, confi = self._getBaseClassifier().partial_dependence(gridGAM, feature=i, width=.9)
            ax.plot(gridGAM[:, i], pdep)
            ax.plot(gridGAM[:, i], confi[0][:, 0], c='grey', ls='--')
            ax.plot(gridGAM[:, i], confi[0][:, 1], c='grey', ls='--')
            ax.set_title(titles[i])
        fig.set_tight_layout(True)
        fig.show()

    def printPlotSummary(self, cv: Union[None, int] = 5):
        ModelNormalAbs.printPlotSummary(self, cv=cv)
        self.plotFeatureFit()


class LogisticLinearLocal(ModelNormalAbs):
    """
    Local Logistic classifier (Linear Proxy)
    Todo: Genuine Local Logistic with weights and add statistics
    """

    def __init__(self, scale=True, reg_type: str = 'll', bw: Union[str, float, Sequence] = 1.):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = skmdl._LogisticLinearLocal(reg_type=reg_type, bw=bw)
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        ModelNormalAbs.__init__(self, model=model, name='Logistic Local (Linear Proxy)')

    def _getBaseClassifier(self) -> skmdl._LogisticLinearLocal:
        return self.model.named_steps['clf']


class LogisticBayesian(ModelNormalAbs):
    """
    Bayesian Logistic
    Todo: Plot feature names in Bayesian LR
    """

    def __init__(self, scale=True, featuresSd=10, nsamplesFit=200, nsamplesPredict=100, mcmc=True,
                 nsampleTune=200, discardTuned=True, samplerStep=None, samplerInit='auto'):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = skmdl._LogisticBayesian(featuresSd=featuresSd, nsamplesFit=nsamplesFit,
                                             nsamplesPredict=nsamplesPredict, mcmc=mcmc,
                                             nsampleTune=nsampleTune, discardTuned=discardTuned,
                                             samplerStep=samplerStep, samplerInit=samplerInit)
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        ModelNormalAbs.__init__(self, model=model, name='Logistic Bayesian')

    def _getBaseClassifier(self) -> skmdl._LogisticBayesian:
        return self.model.named_steps['clf']

    def printCoefficientsInfo(self):
        print('-----Coefficients-----')
        pm.summary(self._getBaseClassifier().trace_)
        print('')

    def plotTrace(self):
        pm.traceplot(self._getBaseClassifier().trace_, varnames=self._getFeatureNames())

    def plotPosterior(self):
        pm.plot_posterior(self._getBaseClassifier().trace_)

    def printPlotSummary(self, cv: Union[None, int] = 5):
        ModelNormalAbs.printPlotSummary(self, cv=cv)
        self.plotPosterior()


class KNN(ModelNormalAbs):
    """
    kNN classifier
    """

    def __init__(self, scale: bool = True, n_neighbors: int = 10, weights: str = 'uniform'):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = sknbr.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric='minkowski')
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        ModelNormalAbs.__init__(self, model=model, name='kNN')

    def _getBaseClassifier(self) -> sknbr.KNeighborsClassifier:
        return self.model.named_steps['clf']


class KNNCV(KNN):
    """
    kNN classifier with CV for k
    *DEPRECIATED*  Use genModelCV instead
    Todo: Check that CV does not give an optimistic score when random states are chained
    """

    def __init__(self, cv=5, scale: bool = True, weights: str = 'uniform'):
        print('*DEPRECIATED*  Use genModelCV instead ')
        self.grid = {'clf__n_neighbors': (5, 10, 20, 40)}
        KNN.__init__(self, scale=scale, n_neighbors=self.grid['clf__n_neighbors'][0], weights=weights)
        self.model = skms.GridSearchCV(self.model, param_grid=self.grid, scoring='accuracy', cv=cv)
        self.name = 'kNN CV'

    def _getBaseClassifier(self) -> sknbr.KNeighborsClassifier:
        return self.model.best_estimator_.named_steps['clf']

    def printBestParamCV(self) -> None:
        print('-----Best CV Parameters-----')
        for param in self.grid:
            print('{} = {:.2f}'.format(param[5:], self.model.best_params_[param]))
        print('...with the score = {:.2f}'.format(self.model.best_score_))
        print('')

    def printCoefficientsInfo(self):
        KNN.printCoefficientsInfo(self)
        self.printBestParamCV()


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

    def _getBaseClassifier(self) -> sktree.DecisionTreeClassifier:
        return self.model.named_steps['clf']

    def printCoefficientsInfo(self):
        print('-----Feature Importance-----')
        importance = self._getBaseClassifier().feature_importances_
        importanceSrs = pd.Series(importance, index=self.x.columns)
        print(importanceSrs)
        print('')

    def visualizeTree(self) -> None:
        dotData = sktree.export_graphviz(self._getBaseClassifier(), precision=2, proportion=True,
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

    def printPlotSummary(self, cv: Union[None, int] = 5):
        ModelNormalAbs.printPlotSummary(self, cv=cv)
        self.visualizeTree()


class TreeCV(Tree):
    """
    Decision Tree classifier with CV for max depth and max number of leaves
    *DEPRECIATED*  Use genModelCV instead
    """

    def __init__(self, cv=5, scale: bool = True, class_weight: Union[None, str] = 'balanced',
                 random_state: Union[int, None] = 1):
        print('*DEPRECIATED*  Use genModelCV instead')
        self.grid = {'clf__max_depth': (2, 3, 4), 'clf__max_leaf_nodes': (4, 6, 8, 12)}
        Tree.__init__(self, scale=scale, max_depth=self.grid['clf__max_depth'][0],
                      class_weight=class_weight, random_state=random_state)
        self.model = skms.GridSearchCV(self.model, param_grid=self.grid, scoring='accuracy', cv=cv)
        self.name = 'Tree CV'

    def _getBaseClassifier(self) -> sktree.DecisionTreeClassifier:
        return self.model.best_estimator_.named_steps['clf']

    def printBestParamCV(self) -> None:
        print('-----Best CV Parameters-----')
        for param in self.grid:
            print('{} = {:.2f}'.format(param[5:], self.model.best_params_[param]))
        print('...with the score = {:.2f}'.format(self.model.best_score_))
        print('')

    def printCoefficientsInfo(self):
        Tree.printCoefficientsInfo(self)
        self.printBestParamCV()


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

    def _getBaseClassifier(self) -> skens.RandomForestClassifier:
        return self.model.named_steps['clf']

    def printCoefficientsInfo(self):
        print('-----Feature Importance-----')
        importance = self._getBaseClassifier().feature_importances_
        importanceSrs = pd.Series(importance, index=self._getFeatureNames())
        print(importanceSrs)
        print('')


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

    def _getBaseClassifier(self) -> skens.GradientBoostingClassifier:
        return self.model.named_steps['clf']

    def printCoefficientsInfo(self):
        print('-----Feature Importance-----')
        importance = self._getBaseClassifier().feature_importances_
        importanceSrs = pd.Series(importance, index=self._getFeatureNames())
        print(importanceSrs)
        print('')

    def plotPartialDependence(self, features=None) -> None:
        """Partial Dependence Plot"""
        if features is None: features = self._getFeatureNames()
        xScaled = self._getScaler().transform(self.x)
        fig, axs = skens.partial_dependence.plot_partial_dependence(self._getBaseClassifier(), X=xScaled,
                                                                    features=features,
                                                                    feature_names=self._getFeatureNames(),
                                                                    percentiles=(0.05, 0.95), grid_resolution=100)
        # fig.suptitle('Partial Dependence: {}'.format(self.name))
        # fig.subplots_adjust(top=0.9)
        fig.set_tight_layout(True)
        fig.show()

    def printPlotSummary(self, cv: Union[None, int] = 5):
        ModelNormalAbs.printPlotSummary(self, cv=cv)
        self.plotPartialDependence()


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
# def printCoefficientsInfo(self):
#     print('-----Feature Importance-----')
#     importance = self._getClassifier().feature_importances_
#     importanceSrs = pd.Series(importance, index=self.x.columns)
#     print(importanceSrs)
#     print('')
#
#
# def plotPartialDependence(self, features=None) -> None:
#     """Partial Dependence Plot"""
#     if features is None: features = self._getFeatureNames()
#     xScaled = self.model.named_steps['scaler'].transform(self.x)
#     fig, axs = skens.partial_dependence.plot_partial_dependence(self._getClassifier(), X=xScaled, features=features,
#                                                                 feature_names=self._getFeatureNames(),
#                                                                 percentiles=(0.05, 0.95), grid_resolution=100)
#     # fig.suptitle('Partial Dependence: {}'.format(self.name))
#     # fig.subplots_adjust(top=0.9)
#     fig.set_tight_layout(True)
#     fig.show()


class SVM(ModelNormalAbs):
    """
    SVM classifier
    """

    def __init__(self, scale: bool = True, C: float = 1., kernel: str = 'poly', degree: int = 2,
                 gamma: Union[str, float] = 'auto',
                 class_weight: Union[None, str] = 'balanced', random_state: Union[int, None] = 1):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = skmdl._SVM(C=C, kernel=kernel, degree=degree, gamma=gamma, class_weight=class_weight,
                                random_state=random_state, probability=False)
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        ModelNormalAbs.__init__(self, model=model, name='SVM ({}):'.format(kernel))

    def _getBaseClassifier(self) -> sksvm.SVC:
        return self.model.named_steps['clf']


class SVMCV(SVM):
    """
    SVM classifier with CV for C
    *DEPRECIATED*  Use genModelCV instead
    """

    def __init__(self, cv=5, scale: bool = True, kernel: str = 'poly', degree: int = 2,
                 class_weight: Union[None, str] = 'balanced', random_state: Union[int, None] = 1):
        print('*DEPRECIATED*  Use genModelCV instead')
        self.grid = {'clf__C': np.exp2(np.arange(-4, 5, 2)), 'clf__gamma': np.exp2(np.arange(-5, -1, 1))}
        SVM.__init__(self, scale=scale, C=self.grid['clf__C'][0], kernel=kernel, degree=degree,
                     gamma=self.grid['clf__gamma'][0], class_weight=class_weight, random_state=random_state)
        self.model = skms.GridSearchCV(self.model, param_grid=self.grid, scoring='accuracy', cv=cv)
        self.name = 'SVM CV ({}):'.format(kernel)

    def _getBaseClassifier(self) -> sksvm.SVC:
        return self.model.best_estimator_.named_steps['clf']

    def printBestParamCV(self) -> None:
        print('-----Best CV Parameters-----')
        for param in self.grid:
            print('{} = {:.2f}'.format(param[5:], self.model.best_params_[param]))
        print('...with the score = {:.2f}'.format(self.model.best_score_))
        print('')

    def printCoefficientsInfo(self):
        SVM.printCoefficientsInfo(self)
        self.printBestParamCV()


class Vote(ModelNormalAbs):
    """
    Voting classifier
    Todo: [Long-term] Make ModelNormalAbs class a decorator for Sklearn-compatible classifiers
    """

    def __init__(self, scale: bool, Models: Sequence[Tuple[str, ModelNormalAbs]], voting='hard', weights=None):
        """
        If scale is True, you can use scale = False for the underlying models. Otherwise, the data will be scaled
        twice.
        """
        self.Models = Models
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = skens.VotingClassifier(estimators=[[x, y.model] for x, y in Models], voting=voting,
                                            weights=weights)
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        ModelNormalAbs.__init__(self, model=model, name='Vote')

    def _getBaseClassifier(self) -> skens.VotingClassifier:
        return self.model.named_steps['clf']

    def printCoefficientsInfo(self):
        for i, Model, estimator in zip(range(len(self.Models)), self.Models, self._getBaseClassifier().estimators_):
            try:
                M = Model[1].copy()
                M.model = estimator
                M._getFeatureNames = lambda: self.x.columns.values
                print('[{} : {}]'.format(i, Model[0]))
                M.printCoefficientsInfo()
            except:
                print('<PRINTING FAILED>')
                print('')


class VoteCV(Vote):
    """
    Voting classifier CV

    This class is more efficient than the class generated by genModelCV
    """

    def __init__(self, cv: int, scale: bool, Models: Sequence[Tuple[str, ModelNormalAbs]], voting='hard',
                 weightsGrid=Sequence[Sequence[float]]):
        """
        If scale is True, you can use scale = False for the underlying models. Otherwise, the data will be scaled
        twice.
        """
        Vote.__init__(self, scale=scale, Models=Models, voting=voting, weights=None)
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = skvote.VotingClassifierCV(estimators=[[x, y.model] for x, y in Models], voting=voting,
                                               weights=weightsGrid, cv=cv, scoring='accuracy')
        self.model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        self.name += ' CV'

    def _getBaseClassifier(self) -> skvote.VotingClassifierCV:
        return self.model.named_steps['clf']

    def printCoefficientsInfo(self):
        Vote.printCoefficientsInfo(self)
        print('-----Best CV Parameters-----')
        print('{} = {}'.format('weights', list(map(utgen.prettyFloat, self._getBaseClassifier().weights_))))
        scores = np.mean(self._getBaseClassifier().scores_, axis=1)
        print('...with the score = {:.2f}   | avg = {:.2f}, std = {:.2f}'
              .format(np.max(scores), np.mean(scores), np.std(scores)))
        print('')


if __name__ == '__main__':
    import theano
    import sklearn

    print('Sklearn v. {}'.format(sklearn.__version__))
    print('PyMC3 v. {}'.format(pm.__version__))
    print('Theano v. {}'.format(theano.__version__))
    print('...device = {}'.format(theano.config.device))
    print('...floatX = {}'.format(theano.config.floatX))
    print('...blas.ldflags = {}'.format(theano.config.blas.ldflags))
    print('...blas.check_openmp = {}'.format(theano.config.blas.check_openmp))
    # theano.test()
    print(os.getcwd())

    model = pm.Model(name='')
    # self.model_.Var('beta', pm.Normal(mu=0, sd=self.featuresSd))
    with model:
        beta = pm.Normal(name='beta', mu=0, sd=4)
    print(model)
