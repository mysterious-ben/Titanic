"""
Auxiliary Sklearn-compatible classification class
"""

from typing import Sequence

import numbers
import os

os.environ['THEANO_FLAGS'] = "floatX=float32"
from typing import Iterable, Sequence, Union

import numpy as np
import pandas as pd
from scipy import optimize
import pygam as gam
from sklearn import base as skbase
import sklearn.model_selection as skms
from sklearn import ensemble as skens
from sklearn import svm as sksvm
from sklearn.utils import validation as skutilvalid
from statsmodels.nonparametric import kernel_regression as smkernel
from statsmodels.nonparametric import _kernel_base as smkernelbase
from sklearn import preprocessing as skprcss
from theano import shared
import pymc3 as pm
from pymc3 import math as pmmath
# import xgboost

import modules.model_framework.utilmodel as utmdl


class _Scaler(skbase.BaseEstimator, skbase.TransformerMixin):
    """Scaler that only applies to selected variables"""

    def __init__(self, copy: bool = True, with_mean: bool = True, with_std: bool = True,
                 features: Union[None, Sequence[int], slice] = None):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.features = features

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        self.scaler_ = skprcss.StandardScaler(copy=self.copy, with_mean=self.with_mean, with_std=self.with_std)
        if isinstance(X, pd.DataFrame): X = X.values  # type: np.ndarray
        if self.features is None: self.features = slice(None)
        self.scaler_.fit(X=X[:, self.features], y=y)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        if isinstance(X, pd.DataFrame): X = X.values  # type: np.ndarray
        X = X.copy().astype(float)  # type: np.ndarray
        X[:, self.features] = self.scaler_.transform(X=X[:, self.features])
        return X


class _LogisticGAM(gam.LogisticGAM):
    """
    Sklearn-compatible Additive Logistic base classifier
    Todo: Imply docstring for modified Sklearn classifiers from the original classifiers
    """

    def __init__(self, verbose=0,
                 lam=0.6, max_iter=100, n_splines=25, spline_order=3,
                 penalties='auto', dtype='auto', tol=1e-4,
                 callbacks=('deviance', 'diffs', 'accuracy'),
                 fit_intercept=True, fit_linear=False, fit_splines=True,
                 constraints=None):
        gam.LogisticGAM.__init__(self, lam=lam, max_iter=max_iter, n_splines=n_splines, spline_order=spline_order,
                                 penalties=penalties, dtype=dtype, tol=tol,
                                 callbacks=callbacks,
                                 fit_intercept=fit_intercept, fit_linear=fit_linear, fit_splines=fit_splines,
                                 constraints=constraints)

    def fit(self, X, y, weights=None):
        self.classes_ = np.unique(y)
        gam.LogisticGAM.fit(self, X=X, y=y, weights=weights)
        return self

    def predict_proba(self, X):
        proba = gam.LogisticGAM.predict_proba(self, X)
        return utmdl.proba2d(proba1d=proba)


class _LogisticLinearLocal(skbase.BaseEstimator, skbase.ClassifierMixin):
    """
    Sklearn-compatible Local Logistic classifier (using Local Linear as a proxy)
    """

    def __init__(self, reg_type: str = 'll', bw: Union[str, float, Iterable] = 'cv_ls'):
        self.reg_type = reg_type
        self.bw = bw

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray, Iterable]) \
            -> skbase.ClassifierMixin:
        X, y = self._check_X_y_fit(X, y)
        self.classes_ = np.unique(y)
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
        dsn_pred = 1 / (1 + np.exp(-dsn_pred))
        return utmdl.proba2d(proba1d=dsn_pred)

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


class _LogisticBayesian(skbase.BaseEstimator, skbase.ClassifierMixin):
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
        return utmdl.proba2d(proba1d=dsn_pred)

    def _check_X_y_fit(self, X, y):
        X, y = skutilvalid.check_X_y(X, y)
        assert np.all(np.unique(y) == np.array([0, 1]))
        return X, y

    def _check_X_predict(self, X):
        X = skutilvalid.check_array(X)
        assert X.shape[1] == self.nfeatures_, "Wrong X shape"
        return X


class _SVM(sksvm.SVC):
    """
    Base Sklearn SVM classifer with a faster (but very approximate) predict_proba function
    """

    def predict_proba(self, X) -> np.ndarray:
        dsn_pred = self.decision_function(X)
        dsn_pred = 1 / (1 + np.exp(-dsn_pred))
        return utmdl.proba2d(proba1d=dsn_pred)


class _VoteRegress(skens.VotingClassifier):
    def __init__(self, estimators: Sequence, voting: str = 'hard',
                 n_jobs: int = 1, flatten_transform=None, cv=None, loss: str = 'square', probaEps: float = 1e-3):
        """
        Sklearn-compatible voting classifier with regressed weights
        """

        skens.VotingClassifier.__init__(self, estimators=estimators, voting=voting,
                                        weights=np.ones(len(estimators)), n_jobs=n_jobs,
                                        flatten_transform=flatten_transform)
        self.cv = cv
        self.loss = loss
        self.probaEps = probaEps

    def fit(self, X, y, sample_weight=None):
        # --fit to the full data
        skens.VotingClassifier.fit(self, X=X, y=y, sample_weight=sample_weight)

        # --generate cross_validated predictions for each classifier
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        if self.voting == 'soft':
            method = 'predict_proba'
        elif self.voting == 'hard':
            method = 'predict'
        else:
            raise LookupError
        predictions = np.empty((X.shape[0], len(self.estimators)), dtype=np.float)
        for i, est in enumerate(self.estimators):
            pred = skms.cross_val_predict(estimator=est[1], X=X, y=y, cv=self.cv, n_jobs=self.n_jobs,
                                          fit_params=fit_params, method=method)
            predictions[:, i] = pred[:, 1]

        # --fit weights using predictions
        X_, y_ = predictions - 0.5, y - 0.5
        nFeatures = X_.shape[1]
        if self.loss == 'square':
            self.weights, _ = optimize.nnls(A=X_, b=y_)
            # self.weights /= np.sum(self.weights)
        elif self.loss == 'squareH':
            bounds = tuple((0, None) for _ in range(nFeatures))
            self.weights = optimize.minimize(fun=utmdl.squareH, x0=np.full(nFeatures, 1. / nFeatures),
                                             args=(X_, y_), bounds=bounds, method='SLSQP')['x']
            # self.weights /= np.sum(self.weights)
        elif self.loss == 'deviance':
            bounds = tuple((0, None) for _ in range(nFeatures))
            constraints = {'type': 'eq', 'fun': lambda x: 1. - np.sum(x)}
            self.weights = optimize.minimize(fun=utmdl.deviance, x0=np.full(nFeatures, 1. / nFeatures),
                                             args=(X_, y_, self.probaEps),
                                             bounds=bounds, constraints=constraints, method='SLSQP')['x']
        elif self.loss == 'devianceNormalized':
            bounds = tuple((0, None) for _ in range(nFeatures))
            constraints = {'type': 'eq', 'fun': lambda x: 1. - np.sum(x)}
            predFactor = (np.max(np.abs(X_), axis=0) * 2) * (1. - self.probaEps)
            X_ /= predFactor[None, :]
            self.weights = optimize.minimize(fun=utmdl.deviance, x0=np.full(nFeatures, 1. / nFeatures),
                                             args=(X_, y_, self.probaEps), bounds=bounds, constraints=constraints,
                                             method='SLSQP')['x']
            self.weights /= predFactor
        else:
            raise LookupError
        return self
