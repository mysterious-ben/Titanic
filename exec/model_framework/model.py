"""
Classification model class

Todo: to add todo-s
"""

from abc import ABC, abstractmethod
from typing import Iterable, Dict, Callable, Tuple

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
import pygam
from sklearn import feature_selection as skfs
from sklearn import linear_model as sklm
from sklearn import metrics as skmtcs
from sklearn import model_selection as skms
from sklearn import neighbors as sknbr
from sklearn import pipeline as skpipe
from sklearn import preprocessing as skprcss
from sklearn import tree as sktree
from sklearn import base as skbase

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

    def score(self, methods: Iterable[str] = ('accuracy',)) -> (Dict[str, float], Dict[str, float], Dict[str, float]):
        """
        Score the classifier

        Args:
            methods: list of score method names

        Returns:
            In-sample, Cross-validated and Out-of-sample scores
        """

        scoreIS = self.scoreIS(methods=methods)
        scoreOOS = self.scoreOOS(methods=methods)
        scoreCV = self.scoreCV(methods=methods)
        return scoreIS, scoreCV, scoreOOS

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
    def scoreCV(self, methods: Iterable[str] = ('accuracy',)) -> Dict[str, float]:
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

    def scoreCV(self, methods: Iterable[str] = ('accuracy',), random_state: int = 1) -> Dict[str, float]:
        scoring = {}
        for method in methods:
            metrics, proba = Metrics.generator(method=method)
            cvScorer = skmtcs.make_scorer(metrics, needs_proba=proba)
            scoring.update({method: cvScorer})

        kfold = skms.KFold(n_splits=5, random_state=random_state)
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

    def printPerformance(self):
        methods = ('accuracy', 'accproba', 'logproba', 'aucproba', 'recall', 'precision')
        scoresIS, scoresCV, scoresOOS = self.score(methods)
        print('-----Performance-----')
        for method in methods:
            print('{}\t (IS / CV / OOS): {:.2f} / {:.2f} / {:.2f}'.format(method, scoresIS[method],
                                                                          scoresCV[method], scoresOOS[method]))
        print('')


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
        assert len(coef) == len(self.x.columns)
        coefSrs = pd.Series(coef, index=self.x.columns)
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


class LogisticGAM(ModelAbs):
    """
    Additive Logistic classifier
    Todo: next
    """

    def __init__(self):
        model = pygam.LogisticGAM()
        ModelAbs.__init__(self, model=model, name='Logistic GAM')


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
        assert len(coef) == len(self.x.columns[support])
        coefSrs = pd.Series(coef, index=self.x.columns[support])
        print('-----Coefficients-----')
        print(coefSrs)
        print('')

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printCoefficients()
        self.printPerformance()
        self.printConfusion()


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
    todo: check that CV does not give an optimistic score
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
    Decision Tree classifier
    """

    def __init__(self, scale: bool = True, max_depth: int = 3, class_weight: str = 'balanced', random_state: int = 1):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = sktree.DecisionTreeClassifier(criterion='gini', max_depth=max_depth, max_leaf_nodes=None,
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
        dotData = sktree.export_graphviz(self._getClassifier(), precision = 2, proportion = True,
                                         feature_names = self.x.columns.values, class_names = ['Dead','Alive'],
                                         impurity = True, filled = True, out_file = None)
        imgPath = 'data\\temp\\tree.png'
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

    def __init__(self, scale=True, class_weight='balanced', random_state=1):
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


if __name__ == '__main__':
    print('Package Model v. 0.1.0')
