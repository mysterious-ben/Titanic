"""
Classification model class
"""

from abc import ABC, abstractmethod

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
from sklearn import feature_selection as skfs
from sklearn import linear_model as sklm
from sklearn import metrics as skmtcs
from sklearn import model_selection as skms
from sklearn import neighbors as sknbr
from sklearn import pipeline as skpipe
from sklearn import preprocessing as skprcss
from sklearn import tree as sktree

import exec.model_framework.utilmodel as utmdl


class Metrics(ABC):
    """
    sklearn-compatible score functions interface
    """
    accuracy = (skmtcs.accuracy_score, False)
    accproba = (lambda y, yH: 1. - skmtcs.mean_absolute_error(y, yH[:, 1]), True)
    logproba = (lambda y, yH: -skmtcs.log_loss(y, yH[:, 1]), True)
    aucproba = (lambda y, yH: skmtcs.roc_auc_score(y, yH[:, 1]), True)
    precision = (skmtcs.precision_score, False)
    recall = (skmtcs.recall_score, False)

    @staticmethod
    def generator(method='accuracy'):
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
    Abstract classifier
    """

    def __init__(self):
        self.model = None
        self.name = None

    @abstractmethod
    def fit(self, data):
        self.x, self.y = utmdl.dataframeToXy(data)
        self.yH = None
        self.yP = None

    @abstractmethod
    def predict(self, data):
        self.xt, self.yt = utmdl.dataframeToXy(data)
        self.ytH = None
        self.ytP = None

    @abstractmethod
    def score(self, methods=('accuracy',)):
        pass

    @abstractmethod
    def scoreIS(self, methods=('accuracy',)):
        pass

    @abstractmethod
    def scoreOOS(self, methods=('accuracy',)):
        pass

    @abstractmethod
    def scoreCV(self, methods=('accuracy',)):
        pass

    def confusionMatrix(self):
        confIS = skmtcs.confusion_matrix(self.y, self.yH)
        confIS = confIS / confIS.sum().sum()
        if self.yt is not None:
            confOSS = skmtcs.confusion_matrix(self.yt, self.ytH)
            confOSS = confOSS / confOSS.sum().sum()
        else:
            confOSS = None
        return confIS, confOSS

    def printSetsInfo(self):
        sampleSize = len(self.y)
        sampleSizeT = len(self.yt) if (self.yt is not None) else None
        posRate = np.sum(self.y) / len(self.y)
        posRateT = np.sum(self.yt) / len(self.yt) if (self.yt is not None) else None
        print('-----Train and Test Sets-----')
        print('Sample Size (Train / Test): {:d} / {:d}'.format(sampleSize, sampleSizeT))
        print('Survived Rate (Train / Test): {:.2f} / {:.2f}'.format(posRate, posRateT))
        print('')

    def printConfusion(self):
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
    def printSummary(self):
        pass

    @abstractmethod
    def plotROC(self):
        pass

    @staticmethod
    def staticPlotROC(y, yP, ax=None, label=' ', title='ROC'):
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
    Abstract sklearn classifier with predict_proba available
    """

    def __init__(self, model, name):
        ModelAbs.__init__(self)
        self.model = model
        self.name = name

    def fit(self, data):
        self.x, self.y = utmdl.dataframeToXy(data)
        self.model.fit(X=self.x, y=self.y)
        self.yH = self.model.predict(X=self.x)
        self.yP = self.model.predict_proba(X=self.x)

    def predict(self, data):
        self.xt, self.yt = utmdl.dataframeToXy(data)
        self.ytH = self.model.predict(X=self.xt)
        self.ytP = self.model.predict_proba(X=self.xt)

    def score(self, methods=('accuracy',)):
        scoreIS = self.scoreIS(methods=methods)
        scoreOOS = self.scoreOOS(methods=methods)
        scoreCV = self.scoreCV(methods=methods)
        return scoreIS, scoreCV, scoreOOS

    def scoreIS(self, methods=('accuracy',)):
        scores = {}
        for method in methods:
            metrics, proba = Metrics.generator(method=method)
            if proba:
                yH = self.yP
            else:
                yH = self.yH
            scores.update({method: metrics(self.y, yH)})
        return scores

    def scoreOOS(self, methods=('accuracy',)):
        scores = {}
        for method in methods:
            metrics, proba = Metrics.generator(method=method)
            if proba:
                ytH = self.ytP
            else:
                ytH = self.ytH
            scores.update({method: metrics(self.yt, ytH) if (self.yt is not None) else None})
        return scores

    def scoreCV(self, methods=('accuracy',), random_state=1):
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

    def printPerformance(self):
        methods = ('accuracy', 'accproba', 'logproba', 'aucproba', 'recall', 'precision')
        scoresIS, scoresCV, scoresOOS = self.score(methods)
        print('-----Performance-----')
        for method in methods:
            print('{}\t (IS / CV / OOS): {:.2f} / {:.2f} / {:.2f}'.format(method, scoresIS[method],
                                                                          scoresCV[method], scoresOOS[method]))
        print('')

    def plotROC(self):
        fig, ax = plt.subplots()
        self.staticPlotROC(self.y, self.yP[:, 1], ax=ax, label='IS', title='ROC: {}'.format(self.name))
        self.staticPlotROC(self.yt, self.ytP[:, 1], ax=ax, label='OOS', title='ROC: {}'.format(self.name))
        fig.set_tight_layout(True)
        fig.show()


class LogisticAbs(ModelNormalAbs):
    """
    Abstract Logistic classifier
    """

    def __init__(self, model, name):
        ModelNormalAbs.__init__(self, model=model, name=name)

    def printCoefficients(self):
        coef = self.model.named_steps['clf'].coef_[0]
        assert len(coef) == len(self.x.columns)
        coefSrs = pd.Series(coef, index=self.x.columns)
        print('-----Coefficients-----')
        print(coefSrs)
        print('')


class Logistic(LogisticAbs):
    """
    Logistic classifier
    """

    def __init__(self, scale=True, fit_intercept=False, C=1.):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = sklm.LogisticRegression(fit_intercept=fit_intercept, C=C, solver='lbfgs', penalty='l2')
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        LogisticAbs.__init__(self, model=model, name='Logistic')

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printCoefficients()
        self.printPerformance()
        self.printConfusion()


class LogisticRidge(LogisticAbs):
    """
    Logistic classifier with Ridge CV penalty
    """

    def __init__(self, scale=True, fit_intercept=False, Cs=10):
        self.name = 'ABC Logistic Regression'
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = sklm.LogisticRegressionCV(fit_intercept=fit_intercept, Cs=Cs, cv=10, solver='lbfgs',
                                               penalty='l2')
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        LogisticAbs.__init__(self, model=model, name='Logistic Ridge')

    def printRidgeMultiplier(self):
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

    def __init__(self, scale=True, fit_intercept=False, k=5, C=1.):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        featureSelector = skfs.SelectKBest(score_func=skfs.f_classif, k=k)
        classifier = sklm.LogisticRegression(fit_intercept=fit_intercept, C=C, solver='lbfgs', penalty='l2')
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('fselect', featureSelector), ('clf', classifier)])
        LogisticAbs.__init__(self, model=model, name='Logistic kBest')

    def printCoefficients(self):
        coef = self.model.named_steps['clf'].coef_[0]
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

    def __init__(self, scale=True, n_neighbors=10, weights='uniform'):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = sknbr.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric='minkowski')
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        ModelNormalAbs.__init__(self, model=model, name='kNN')

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printPerformance()
        self.printConfusion()


class Tree(ModelNormalAbs):
    """
    Decision Tree classifier
    """

    def __init__(self, scale=True, max_depth=3, class_weight='balanced', random_state=1):
        scaler = skprcss.StandardScaler(with_mean=scale, with_std=scale)
        classifier = sktree.DecisionTreeClassifier(criterion='gini', max_depth=max_depth, class_weight=class_weight,
                                                   splitter='best', min_samples_leaf=5, random_state=random_state)
        model = skpipe.Pipeline(steps=[('scaler', scaler), ('clf', classifier)])
        ModelNormalAbs.__init__(self, model=model, name='Decision Tree')

    def printSummary(self):
        print('****** {} ******\n'.format(str.upper(self.name)))
        self.printSetsInfo()
        self.printPerformance()
        self.printConfusion()

    def visualizeTree(self):
        dotData = sktree.export_graphviz(self.model.named_steps['clf'], precision=2, proportion=True,
                                         feature_names=self.x.columns.values, class_names=['Dead', 'Alive'],
                                         impurity=True, filled=True, out_file=None)
        imgPath = 'data\\temp\\tree.png'
        pydotplus.graph_from_dot_data(dotData).write_png(imgPath)
        image = img.imread(imgPath)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title('Graph: {}'.format(self.name))
        fig.set_tight_layout(True)
        fig.show()


if __name__ == '__main__':
    print('Package Model v. 0.1.0')