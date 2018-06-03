"""
Classification model class
"""

from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Dict, Callable, Union, Type
import warnings
import inspect
import copy
import os

os.environ['THEANO_FLAGS'] = "floatX=float32"

import matplotlib.image as img
import matplotlib.pyplot as plt
import seaborn as sns
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
# from sklearn import preprocessing as skprcss
from sklearn import tree as sktree
from sklearn import svm as sksvm
import pymc3 as pm
# import xgboost

import modules.data_framework.utildata as utdata
import modules.model_framework.utilmodel as utmdl
import modules.model_framework.sklearn_model as skmdl
import modules.model_framework.voting_classifier_cv as skvote

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


class Metrics(ABC):
    """
    Score method interface (compatible with Scikit-learn)
    """

    accuracy = (skmtcs.accuracy_score, False)
    accproba = (lambda y, yH: 1. - skmtcs.mean_absolute_error(y, yH[:, 1]), True)
    logproba = (lambda y, yH: -skmtcs.log_loss(y, yH[:, 1], labels=[0, 1]), True)
    aucproba = (lambda y, yH: skmtcs.roc_auc_score(y, yH[:, 1]), True)
    precision = (lambda y, yH: skmtcs.precision_score(y, yH, labels=[0, 1]), False)
    recall = (lambda y, yH: skmtcs.recall_score(y, yH, labels=[0, 1]), False)

    @staticmethod
    def generator(method: str = 'accuracy') -> Tuple[Callable[[Sequence[float], Sequence[float]], float], bool]:
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
        x:
        xt:
        y:
        yt:
        yH:
        ytH:
        yP:
        ytP:
    """

    name = 'Abstract Classifier'

    def __init__(self, scale: str):
        self.scale = scale

    # @staticmethod
    # def _assignInitArgs(obj, frame):
    #     args, _, _, values = inspect.getargvalues(frame)
    #     values.pop("self")
    #     for arg, val in values.items():
    #         setattr(obj, arg, val)

    def copy(self):
        return copy.deepcopy(self)

    def _makeScaler(self) -> skmdl._Scaler:
        # return skprcss.StandardScaler(with_mean=self.scale, with_std=self.scale)
        if self.scale == 'none':
            meanstd = False
            scaleFeatures = None  # list(range(len(self.x.columns)))
        elif self.scale == 'all':
            meanstd = True
            scaleFeatures = None  # list(range(len(self.x.columns)))
        elif self.scale == 'some':
            meanstd = True
            scaleFeatures = utdata.scaleFeatureIndices(data=self.x)
        else:
            raise KeyError
        return skmdl._Scaler(copy=True, with_mean=meanstd, with_std=meanstd, features=scaleFeatures)

    @abstractmethod
    def _makeBaseClassifier(self) -> skbase.ClassifierMixin:
        pass

    def _makeModel(self) -> skpipe.Pipeline:
        assert hasattr(self, 'x') and hasattr(self, 'y'), "No self.x and self.y: Data needs beforehand"
        scaler = self._makeScaler()
        classifier = self._makeBaseClassifier()
        return skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])

    def _getScaler(self) -> skmdl._Scaler:
        """Return Sklearn data scaler"""
        return self.model.named_steps['scaler']

    @abstractmethod
    def _getBaseClassifier(self) -> skbase.ClassifierMixin:
        """Return Sklearn base classifier"""
        return self.model.named_steps['clf']

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the model to the data (features + labels)

        Args:
            data: DataFrame with **x** and **y**
            scale: If true, normalize features
        """

        self.x, self.y = utmdl.dataframeToXy(data)
        self.model = self._makeModel()
        self.model.fit(X=self.x, y=self.y)
        self.yH = self.model.predict(X=self.x)
        try:
            self.yP = self.model.predict_proba(X=self.x)
        except AttributeError:
            self.yP = None

    def _getFeatureNames(self):
        """Return names of the original features"""
        return self.x.columns.values

    @staticmethod
    def balancedWeights(y):
        posRate = np.sum(y) / len(y)
        weights = y * (1. - posRate) + (1. - y) * posRate
        return weights

    def predict(self, data: pd.DataFrame) -> None:
        """
        Predict labels for the chosen features

        Args:
            data: DataFrame with **xt** (and **yt**, optionally)
        """

        self.xt, self.yt = utmdl.dataframeToXy(data)
        self.ytH = self.model.predict(X=self.xt)
        try:
            self.ytP = self.model.predict_proba(X=self.xt)
        except AttributeError:
            self.ytP = None

    def submission(self) -> pd.DataFrame:
        """Create DataFrame for Kaggle submission using ytH"""

        submission = pd.DataFrame(index=self.xt.index, columns=['Survived'])
        submission.loc[:, 'Survived'] = self.ytH
        return submission

    def fitPredict(self, dataTrain: pd.DataFrame, dataTest: pd.DataFrame) -> None:
        """Fit, then predict"""

        self.fit(data=dataTrain)
        self.predict(data=dataTest)

    def fitPredictSubmission(self, dataTrain: pd.DataFrame, dataTest: pd.DataFrame) -> pd.DataFrame:
        """Fit, then predict, then create a submission"""

        self.fit(data=dataTrain)
        self.predict(data=dataTest)
        # print('Predicted survival rate in the submission = {:.2f}'.format(np.sum(self.ytH) / len(self.ytH)))
        return self.submission()

    def _ytValid(self) -> bool:
        return (self.yt is not None) and (np.sum(np.isnan(self.yt)) == 0) and (0 in self.yt) and (1 in self.yt)

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

    def scoreCV(self, methods: Sequence[str] = ('accuracy',), cv: int = 5, random_state: Union[None, int] = None) \
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

    def confusionMatrix(self) -> (np.ndarray, np.ndarray):
        """Compute a confusion matrix of the classifier (IS and OOS)"""
        confIS = skmtcs.confusion_matrix(self.y, self.yH)
        confIS = confIS / confIS.sum().sum()
        if self._ytValid():
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
        posRatePred = np.sum(self.yH) / len(self.yH)
        posRateT = np.sum(self.yt) / len(self.yt) if (self._ytValid()) else 0
        posRatePredT = np.sum(self.ytH) / len(self.ytH)
        print('-----Train and Test Sets-----')
        print('Sample Size (Train / Test): {:d} / {:d}'.format(sampleSize, sampleSizeT))
        print('Train Survived Rate (True / Predicted): {:.2f} / {:.2f}'.format(posRate, posRatePred))
        print('Test Survived Rate (True / Predicted): {:.2f} / {:.2f}'.format(posRateT, posRatePredT))
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
            methods = ('accuracy', 'logproba', 'aucproba', 'recall', 'precision')  # 'accproba',
        scoreIS = self.scoreIS(methods)
        scoreOOS = self.scoreOOS(methods) if self._ytValid() else {x: np.nan for x in methods}
        scoreCV = self.scoreCV(methods, cv=cv) if cv is not None else {x: np.nan for x in methods}
        print('-----Performance-----')
        for method in methods:
            print('{}\t (IS / CV / OOS): {:.2f} / {:.2f} ({:.2f}) / {:.2f}'.
                  format(method, scoreIS[method], np.mean(scoreCV[method]), np.std(scoreCV[method]), scoreOOS[method]))
        print('')

    def printCoefficientsInfo(self):
        """Print information on classifier coefficients"""
        print()

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
        elif not self._ytValid():
            print('<TEST OUTCOMES ARE NOT VALID: ROC PLOT IS NOT AVAILABLE>')
        else:
            fig, ax = plt.subplots(figsize=(6.5, 6))
            self.staticPlotROC(self.y, self.yP[:, 1], ax=ax, label='IS', title='ROC: {}'.format(self.name))
            self.staticPlotROC(self.yt, self.ytP[:, 1], ax=ax, label='OOS', title='ROC: {}'.format(self.name))
            # fig.set_tight_layout(True)
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


def genModelCV(ModelClass: Type[ModelAbs], cv: int, grid: Dict[str, Sequence]):
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
        name = ModelClass.name + ' CV'

        def __init__(self, *args, **kwargs):
            ModelClass.__init__(self, *args, **kwargs)
            assert not hasattr(self, 'grid'), 'ModelCV error: base model already has *grid* attribute'
            assert not hasattr(self, 'cv'), 'ModelCV error: base model already has *cv* attribute'
            self.grid = {'clf__' + x: grid[x] for x in grid}
            self.cv = cv

        def _makeModel(self) -> skms.GridSearchCV:
            model = ModelClass._makeModel(self)
            return skms.GridSearchCV(estimator=model, param_grid=self.grid, scoring='accuracy', cv=self.cv)

        def _getBaseClassifier(self):
            return self.model.best_estimator_.named_steps['clf']

        def _getScaler(self):
            return self.model.best_estimator_.named_steps['scaler']

        def printBestParamCV(self) -> None:
            print('-----Best CV Parameters-----')
            for param in self.grid:
                print('{} = {:.2f}'.format(param[5:], self.model.best_params_[param]))
            scores = self.model.cv_results_['mean_test_score']
            print('...with the score = {:.2f}   | avg = {:.2f}, std = {:.2f}'
                  .format(np.max(scores), np.mean(scores), np.std(scores)))
            print('')

        def printCoefficientsInfo(self) -> None:
            ModelClass.printCoefficientsInfo(self)
            self.printBestParamCV()

    return ModelCV


class LogisticAbs(ModelAbs):
    """
    Abstract Logistic classifier
    """

    name = 'Abstract Logistic'

    def __init__(self, scale: str):
        ModelAbs.__init__(self, scale=scale)

    @abstractmethod
    def _getBaseClassifier(self) -> sklm.LogisticRegression:
        return self.model.named_steps['clf']

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
    Todo: Add Z-score statistics
    """

    name = 'Logistic'

    def __init__(self, scale: str = 'some', fit_intercept: bool = False, C: int = 1., penalty='l2'):
        LogisticAbs.__init__(self, scale=scale)
        self.fit_intercept = fit_intercept
        self.C = C
        self.penalty = penalty

    def _makeBaseClassifier(self) -> sklm.LogisticRegression:
        return sklm.LogisticRegression(fit_intercept=self.fit_intercept, C=self.C, penalty=self.penalty,
                                       solver='lbfgs')

    def _getBaseClassifier(self) -> sklm.LogisticRegression:
        return self.model.named_steps['clf']


class LogisticRidgeCV(LogisticAbs):
    """
    Logistic classifier with Ridge CV penalty
    Todo: Logistic Ridge with a specified number of degrees of freedom
    """

    name = 'Logistic Ridge'

    def __init__(self, scale: str = 'some', fit_intercept: bool = False, Cs: int = 10, penalty='l2'):
        LogisticAbs.__init__(self, scale=scale)
        self.fit_intercept = fit_intercept
        self.Cs = Cs
        self.penalty = penalty

    def _makeBaseClassifier(self) -> sklm.LogisticRegressionCV:
        return sklm.LogisticRegressionCV(fit_intercept=self.fit_intercept, Cs=self.Cs, cv=10, penalty=self.penalty,
                                         solver='lbfgs')

    def _getBaseClassifier(self) -> sklm.LogisticRegressionCV:
        return self.model.named_steps['clf']

    def printCoefficientsInfo(self) -> None:
        LogisticAbs.printCoefficientsInfo(self)
        print('-----Ridge CV multiplier-----')
        print('Ridge Multiplier (inverse reg. strength) = {:.2f}'.format(self.model.named_steps['clf'].C_[0]))
        print('')


class LogisticBestSubset(LogisticAbs):
    """
    Logistic classifier with k features pre-selected
    """

    name = 'Logistic kBest'

    def __init__(self, scale: str = 'some', fit_intercept: bool = False, k: int = 5, C: float = 1.,
                 selectfun='f_classif', penalty='l2'):
        LogisticAbs.__init__(self, scale=scale)
        self.fit_intercept = fit_intercept
        self.k = k
        self.C = C
        self.selectfun = selectfun
        self.penalty = penalty

    def _makeBaseClassifier(self) -> sklm.LogisticRegression:
        return sklm.LogisticRegression(fit_intercept=self.fit_intercept, C=self.C, penalty=self.penalty,
                                       solver='lbfgs')

    def _getBaseClassifier(self) -> sklm.LogisticRegression:
        return self.model.named_steps['clf']

    def _makeSelector(self):
        if self.selectfun == 'f_classif':
            fun = skfs.f_classif
        else:
            raise LookupError
        return skfs.SelectKBest(score_func=fun, k=self.k)

    def _makeModel(self) -> skpipe.Pipeline:
        scaler = self._makeScaler()
        featureSelector = self._makeSelector()
        classifier = self._makeBaseClassifier()
        return skpipe.Pipeline(steps=[('scaler', scaler), ('fselect', featureSelector), ('clf', classifier)])

    def printCoefficientsInfo(self):
        coef = self._getBaseClassifier().coef_[0]
        support = self.model.named_steps['fselect'].get_support()
        assert len(coef) == len(self._getFeatureNames()[support])
        coefSrs = pd.Series(coef, index=self._getFeatureNames()[support])
        print('-----Coefficients-----')
        print(coefSrs)
        print('')


class LogisticGAM(ModelAbs):
    """
    Additive Logistic classifier
    """

    name = 'Logistic GAM'

    def __init__(self, scale=True, fit_intercept=False, n_splines=15, lam=1., constraints=None):
        ModelAbs.__init__(self, scale=scale)
        self.fit_intercept = fit_intercept
        self.n_splines = n_splines
        self.lam = lam
        self.constraints = constraints

    def _makeBaseClassifier(self) -> gam.LogisticGAM:
        return skmdl._LogisticGAM(fit_intercept=self.fit_intercept, n_splines=self.n_splines, lam=self.lam,
                                  constraints=self.constraints)

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

        N_COLS = 5
        gridGAM = gamutils.generate_X_grid(self._getBaseClassifier())
        # plt.rcParams['figure.figsize'] = (28, 8)
        nAxRow, nAxCol = len(self._getFeatureNames()) // N_COLS + 1, N_COLS
        fig, axs = plt.subplots(nrows=nAxRow, ncols=nAxCol, figsize=(14, 3 * nAxRow))
        titles = self._getFeatureNames()
        for i in range(len(self._getFeatureNames())):
            pdep, confi = self._getBaseClassifier().partial_dependence(gridGAM, feature=i, width=.9)
            axIdx = np.unravel_index(i, (nAxRow, nAxCol))
            axs[axIdx[0], axIdx[1]].plot(gridGAM[:, i], pdep)
            axs[axIdx[0], axIdx[1]].plot(gridGAM[:, i], confi[0][:, 0], c='grey', ls='--')
            axs[axIdx[0], axIdx[1]].plot(gridGAM[:, i], confi[0][:, 1], c='grey', ls='--')
            axs[axIdx[0], axIdx[1]].set_title(titles[i])
        # fig.set_tight_layout(True)
        fig.show()

    def printPlotSummary(self, cv: Union[None, int] = 5):
        ModelAbs.printPlotSummary(self, cv=cv)
        self.plotFeatureFit()


class LogisticLinearLocal(ModelAbs):
    """
    Local Logistic classifier (Linear Proxy)
    Todo: Genuine Local Logistic with weights and add statistics
    """

    name = 'Linear Local'

    def __init__(self, scale=True, reg_type: str = 'll', bw: Union[str, float, Sequence] = 1.):
        ModelAbs.__init__(self, scale=scale)
        self.reg_type = reg_type
        self.bw = bw

    def _makeBaseClassifier(self) -> skmdl._LogisticLinearLocal:
        return skmdl._LogisticLinearLocal(reg_type=self.reg_type, bw=self.bw)

    def _getBaseClassifier(self) -> skmdl._LogisticLinearLocal:
        return self.model.named_steps['clf']


class LogisticBayesian(ModelAbs):
    """
    Bayesian Logistic
    Todo: Plot feature names in Bayesian LR
    """

    name = 'Logistic Bayesian'

    def __init__(self, scale=True, featuresSd=10, nsamplesFit=200, nsamplesPredict=100, mcmc=True,
                 nsampleTune=200, discardTuned=True, samplerStep=None, samplerInit='auto'):
        ModelAbs.__init__(self, scale=scale)
        self.featuresSd = featuresSd
        self.nsamplesFit = nsamplesFit
        self.nsamplesPredict = nsamplesPredict
        self.mcmc = mcmc
        self.nsampleTune = nsampleTune
        self.discardTuned = discardTuned
        self.samplerStep = samplerStep
        self.samplerInit = samplerInit

    def _makeBaseClassifier(self) -> skmdl._LogisticBayesian:
        return skmdl._LogisticBayesian(featuresSd=self.featuresSd, nsamplesFit=self.nsamplesFit,
                                       nsamplesPredict=self.nsamplesPredict, mcmc=self.mcmc,
                                       nsampleTune=self.nsampleTune, discardTuned=self.discardTuned,
                                       samplerStep=self.samplerStep, samplerInit=self.samplerInit)

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
        ModelAbs.printPlotSummary(self, cv=cv)
        self.plotPosterior()


class KNN(ModelAbs):
    """
    kNN classifier
    """

    name = 'kNN'

    def __init__(self, scale: str = 'some', n_neighbors: int = 10, weights: str = 'uniform', metric: str = 'minkowski'):
        ModelAbs.__init__(self, scale=scale)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

    def _makeBaseClassifier(self) -> sknbr.KNeighborsClassifier:
        return sknbr.KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, metric=self.metric)

    def _getBaseClassifier(self) -> sknbr.KNeighborsClassifier:
        return self.model.named_steps['clf']


class KNNCV(KNN):
    """
    kNN classifier with CV for k
    *DEPRECIATED*  Use genModelCV instead
    Todo: Check that CV does not give an optimistic score when random states are chained
    """

    name = 'kNN CV'

    def __init__(self, cv=5, scale: str = 'some', weights: str = 'uniform', grid=None):
        warnings.warn('*DEPRECIATED*  Use genModelCV instead', DeprecationWarning)
        if grid is None:
            self.grid = {'clf__n_neighbors': (5, 10, 20, 40)}
        else:
            self.grid = {'clf__' + x: grid[x] for x in grid}
        KNN.__init__(self, scale=scale, n_neighbors=self.grid['clf__n_neighbors'][0], weights=weights)
        self.cv = cv

    def _makeModel(self) -> skms.GridSearchCV:
        model = KNN._makeModel(self)
        return skms.GridSearchCV(estimator=model, param_grid=self.grid, scoring='accuracy', cv=self.cv)

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


class Tree(ModelAbs):
    """
    Decision Tree classifier (CART algorithm)
    """

    name = 'Decision Tree'

    def __init__(self, scale: str = 'some', max_depth: Union[int, None] = 3, max_leaf_nodes: Union[int, None] = None,
                 class_weight: Union[None, str] = 'balanced', criterion='gini', min_samples_leaf=5,
                 random_state: Union[None, int] = None):
        ModelAbs.__init__(self, scale=scale)
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.class_weight = class_weight
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _makeBaseClassifier(self) -> sktree.DecisionTreeClassifier:
        return sktree.DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth,
                                             max_leaf_nodes=self.max_leaf_nodes,
                                             class_weight=self.class_weight, min_samples_leaf=self.min_samples_leaf,
                                             random_state=self.random_state)

    def _getBaseClassifier(self) -> sktree.DecisionTreeClassifier:
        return self.model.named_steps['clf']

    def printCoefficientsInfo(self):
        print('-----Feature Importance-----')
        importance = self._getBaseClassifier().feature_importances_
        importanceSrs = pd.Series(importance, index=self._getFeatureNames())
        print(importanceSrs)
        print('')

    def visualizeTree(self) -> None:
        dotData = sktree.export_graphviz(self._getBaseClassifier(), precision=2, proportion=True,
                                         feature_names=self._getFeatureNames(), class_names=['Dead', 'Alive'],
                                         impurity=True, filled=True, out_file=None)
        dotData = (dotData[:15] +
                   'graph [ dpi = 400 ]; \n' +
                   'graph[fontname = "helvetica"]; \n' +
                   'node[fontname = "helvetica"]; \n' +
                   'edge[fontname = "helvetica"]; \n' +
                   dotData[15:])
        imgPath = os.path.join('data', 'temp', 'tree.png')
        pydotplus.graph_from_dot_data(dotData).write_png(imgPath)
        image = img.imread(imgPath)
        fig, ax = plt.subplots(figsize=(14, 14))
        fig.set_tight_layout(True)
        ax.imshow(image)
        ax.set_title('Graph: {}'.format(self.name), fontsize=14)
        ax.axis('off')
        fig.show()

    def printPlotSummary(self, cv: Union[None, int] = 5):
        ModelAbs.printPlotSummary(self, cv=cv)
        self.visualizeTree()


class RandomForest(ModelAbs):
    """
    Random Forest classifier (Decision Trees + Bagging)
    Todo: Add OOB training progress plot
    """

    name = 'Random Forest'

    def __init__(self, scale: str = 'some', n_estimators: int = 128, max_features: Union[int, None] = None,
                 max_depth: Union[int, None] = None, max_leaf_nodes: Union[int, None] = 12,
                 class_weight: Union[None, str] = 'balanced', criterion='gini', min_samples_leaf=5,
                 random_state: Union[None, int] = None):
        ModelAbs.__init__(self, scale=scale)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.criterion = criterion
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _makeBaseClassifier(self) -> skens.RandomForestClassifier:
        return skens.RandomForestClassifier(n_estimators=self.n_estimators, max_features=self.max_features,
                                            max_leaf_nodes=self.max_leaf_nodes, criterion=self.criterion,
                                            max_depth=self.max_depth, class_weight=self.class_weight,
                                            min_samples_leaf=self.min_samples_leaf, random_state=self.random_state)

    def _getBaseClassifier(self) -> skens.RandomForestClassifier:
        return self.model.named_steps['clf']

    def printCoefficientsInfo(self):
        print('-----Feature Importance-----')
        COEF_THRESHOLD = 0.01
        importance = self._getBaseClassifier().feature_importances_
        importanceSrs = pd.Series(importance, index=self._getFeatureNames()).sort_values(ascending=False)
        importanceSrs = importanceSrs.loc[importanceSrs > COEF_THRESHOLD]
        print(importanceSrs)
        print('')


class BoostedTree(ModelAbs):
    """
    Boosted Trees (Gradient Boosting)
    """

    name = 'Boosted Tree'

    def __init__(self, scale: str = 'some', n_estimators: int = 128, loss: str = 'deviance', learning_rate: float = 1.,
                 subsample: float = 1., max_features: Union[int, None] = None,
                 max_depth: Union[int, None] = 2, max_leaf_nodes: Union[int, None] = None,
                 min_samples_leaf=5,
                 random_state: Union[None, int] = None, class_weight: Union[None, str] = None):
        ModelAbs.__init__(self, scale=scale)
        self.n_estimators = n_estimators
        self.loss = loss
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.class_weight = class_weight

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

    def _makeBaseClassifier(self) -> skens.GradientBoostingClassifier:
        if self.class_weight is None:
            classifierClass = skens.GradientBoostingClassifier
        elif self.class_weight == 'balanced':
            class GradientBoostingClassifierBalanced(skens.GradientBoostingClassifier):
                def fit(self, X, y, sample_weight=None, monitor=None):
                    weights = ModelAbs.balancedWeights(y)
                    return skens.GradientBoostingClassifier.fit(self, X=X, y=y, sample_weight=weights)

            classifierClass = GradientBoostingClassifierBalanced
        else:
            raise LookupError

        return classifierClass(n_estimators=self.n_estimators, loss=self.loss,
                               learning_rate=self.learning_rate, subsample=self.subsample,
                               max_features=self.max_features, max_leaf_nodes=self.max_leaf_nodes,
                               max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                               random_state=self.random_state)

    def _getBaseClassifier(self) -> skens.GradientBoostingClassifier:
        return self.model.named_steps['clf']

    def printCoefficientsInfo(self):
        print('-----Feature Importance-----')
        COEF_THRESHOLD = 0.01
        importance = self._getBaseClassifier().feature_importances_
        importanceSrs = pd.Series(importance, index=self._getFeatureNames()).sort_values(ascending=False)
        importanceSrs = importanceSrs.loc[importanceSrs > COEF_THRESHOLD]
        print(importanceSrs)
        print('')

    def plotPartialDependence(self, features=None) -> None:
        """Partial Dependence Plot"""
        if features is None: features = self._getFeatureNames()
        xScaled = self._getScaler().transform(self.x)
        fig, axs = skens.partial_dependence.plot_partial_dependence(self._getBaseClassifier(), X=xScaled,
                                                                    features=features,
                                                                    feature_names=self._getFeatureNames(),
                                                                    percentiles=(0.05, 0.95), grid_resolution=100,
                                                                    n_cols=5)
        # fig.suptitle('Partial Dependence: {}'.format(self.name))
        # fig.subplots_adjust(top=0.9)
        fig.set_size_inches(14, 3 * (len(self._getFeatureNames()) // 5))
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        fig.suptitle('Partial dependence: {}'.format(self.name))
        fig.show()

    def printPlotSummary(self, cv: Union[None, int] = 5):
        ModelAbs.printPlotSummary(self, cv=cv)
        self.plotPartialDependence()


class BoostedTreeXGBoost(ModelAbs):
    """
    Boosted Trees (XGBoost)
    UNDER DEVELOPMENT
    Todo: Complete BoostedTreeXGBoost (fix import xgboost and add balanced weight scaling)
    """
    pass


#     def __init__(self, scale: str = 'some', n_estimators: int = 128, loss: str = 'deviance', learning_rate: float = 1.,
#                  subsample: int = 1., max_features: Union[int, None] = None,
#                  max_depth: Union[int, None] = 2, max_leaf_nodes: Union[int, None] = None,
#                  random_state: Union[None, int] = None, balanceWeights: bool = False):
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
#         ModelAbs.__init__(self, model=model, name='Boosted Tree (XGBoost)')
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


class SVM(ModelAbs):
    """
    SVM classifier
    """

    name = 'SVM'

    def __init__(self, scale: str = 'some', C: float = 1., kernel: str = 'poly', degree: int = 2,
                 gamma: Union[str, float] = 'auto',
                 class_weight: Union[None, str] = 'balanced', random_state: Union[int, None] = None):
        ModelAbs.__init__(self, scale=scale)
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = random_state

    def _makeBaseClassifier(self) -> sksvm.SVC:
        return skmdl._SVM(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                          class_weight=self.class_weight, random_state=self.random_state,
                          probability=False)

    def _getBaseClassifier(self) -> sksvm.SVC:
        return self.model.named_steps['clf']


class Vote(ModelAbs):
    """
    Voting classifier
    Todo: Change the hacky way to create Vote classifier
    Todo: [Long-term] Make ModelAbs class a decorator for Sklearn-compatible classifiers
    """

    name = 'Vote'

    def __init__(self, scale: str, Models: Sequence[Tuple[str, ModelAbs]], voting='hard', weights=None,
                 baseClassifiersInfo=True):
        """
        If scale is True, you can use scale = False for the underlying models. Otherwise, the data will be scaled
        twice.
        """
        ModelAbs.__init__(self, scale=scale)
        self.Models = Models
        self.voting = voting
        self.weights = weights
        self.baseClassifiersInfo = baseClassifiersInfo

    def _makeBaseClassifier(self) -> skens.VotingClassifier:
        return skens.VotingClassifier(estimators=[[x, y._makeModel()] for x, y in self.Models], voting=self.voting,
                                      weights=self.weights)

    def _getBaseClassifier(self) -> skens.VotingClassifier:
        return self.model.named_steps['clf']

    def fit(self, data):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module='sklearn')
            self.x, self.y = utmdl.dataframeToXy(data)
            for basename, basemodel in self.Models:
                basemodel.x = self.x
                basemodel.y = self.y
            self.model = self._makeModel()
            self.model.fit(X=self.x, y=self.y)
            self.yH = self.model.predict(X=self.x)
            try:
                self.yP = self.model.predict_proba(X=self.x)
            except AttributeError:
                self.yP = None

    def predict(self, data):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module='sklearn')
            ModelAbs.predict(self, data=data)

    def printSummary(self, cv: Union[None, int] = 5):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module='sklearn')
            ModelAbs.printSummary(self, cv=cv)

    def printCoefficientsInfo(self):
        if self.baseClassifiersInfo:
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
        else:
            ModelAbs.printCoefficientsInfo(self)


class VoteCV(Vote):
    """
    Voting classifier with the weights found by CV

    This class is more efficient than the class generated by genModelCV
    """

    name = 'Vote CV'

    def __init__(self, weightsCv: int, weightsGrid: Sequence[Sequence[float]],
                 scale: str, Models: Sequence[Tuple[str, ModelAbs]], voting='hard',
                 baseClassifiersInfo=True):
        """If scale is True, you can use scale = False for the underlying models. Otherwise, the data will be scaled
        twice."""

        Vote.__init__(self, scale=scale, Models=Models, voting=voting, weights=None,
                      baseClassifiersInfo=baseClassifiersInfo)
        self.weightsCv = weightsCv
        self.weightsGrid = weightsGrid

    def _makeBaseClassifier(self) -> skvote.VotingClassifierCV:
        return skvote.VotingClassifierCV(estimators=[[x, y._makeModel()] for x, y in self.Models], voting=self.voting,
                                         weights=self.weightsGrid, cv=self.weightsCv, scoring='accuracy')

    def _getBaseClassifier(self) -> skvote.VotingClassifierCV:
        return self.model.named_steps['clf']

    def printCoefficientsInfo(self):
        Vote.printCoefficientsInfo(self)
        print('-----Best CV Parameters-----')
        w = np.array(self._getBaseClassifier().weights_)
        idx = np.argsort(-w)
        print('-----Weights-----')
        for i in idx:
            if w[i] > 0: print('{} = {:.2f}'.format(self.Models[i][0], w[i]))
        scores = np.mean(self._getBaseClassifier().scores_, axis=1)
        print('...with the score = {:.2f}   | avg = {:.2f}, std = {:.2f}'
              .format(np.max(scores), np.mean(scores), np.std(scores)))
        print('')


class VoteRegress(Vote):
    """
    Voting classifier the weights found by regression

    This class is much faster than VoteCV for a larger number of base classifiers (>5)
    """

    name = 'Vote Regress'

    def __init__(self, weightsCv: int,
                 scale: str, Models: Sequence[Tuple[str, ModelAbs]], voting='hard',
                 loss='square', baseClassifiersInfo=True):
        """If scale is True, you can use scale = False for the underlying models. Otherwise, the data will be scaled
        twice."""

        Vote.__init__(self, scale=scale, Models=Models, voting=voting, weights=None,
                      baseClassifiersInfo=baseClassifiersInfo)
        self.weightsCv = weightsCv
        self.loss = loss

    def _makeBaseClassifier(self) -> skmdl._VoteRegress:
        return skmdl._VoteRegress(estimators=[[x, y._makeModel()] for x, y in self.Models], voting=self.voting,
                                  cv=self.weightsCv, loss=self.loss)

    def _getBaseClassifier(self) -> skmdl._VoteRegress:
        return self.model.named_steps['clf']

    def printCoefficientsInfo(self):
        COEF_THRESHOLD = 0.01
        Vote.printCoefficientsInfo(self)
        w = self._getBaseClassifier().weights
        idx = np.argsort(-w)
        print('-----Weights-----')
        for i in idx:
            if w[i] > COEF_THRESHOLD: print('{} = {:.2f}'.format(self.Models[i][0], w[i]))
        print('')

    def plotPredictCorrelations(self):
        df = pd.DataFrame(self._getBaseClassifier().predictions_, columns=[x[0] for x in self.Models])
        fig, ax = plt.subplots(figsize=(8, 7))
        g = sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
        ax.set_title('Correlations')
        fig.tight_layout()

    def printPlotSummary(self, cv: Union[None, int] = 5):
        ModelAbs.printPlotSummary(self, cv=cv)
        self.plotPredictCorrelations()



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
    # theano.tests()
    print(os.getcwd())

    model = pm.Model(name='')
    # self.model_.Var('beta', pm.Normal(mu=0, sd=self.featuresSd))
    with model:
        beta = pm.Normal(name='beta', mu=0, sd=4)
    print(model)
