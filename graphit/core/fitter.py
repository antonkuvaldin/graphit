from abc import ABC, abstractmethod
from typing import Optional
from . import Embedding, Regressor, Optimizer, ProgressBar, Report

import copy
import numpy


class Fitter(ABC):
    def __init__(self, embedding: Embedding,
                 regressor: Regressor,
                 optimizer: Optional[Optimizer] = None,
                 name: Optional[str] = 'model',
                 pbar: Optional[ProgressBar] = None):
        self.name = name
        self.embedding = copy.deepcopy(embedding)
        self.regressor = copy.deepcopy(regressor)
        self.emb_study = None
        self.regr_study = None
        self.optimizer = optimizer
        self.pbar = pbar

    def init_pbar(self, samples_num: int, test_size, report: Report):
        self.embedding.set_pbar(self.pbar)
        report.set_pbar(self.pbar)
        metrics_num = len(report._metrics)
        if self.pbar is None:
            return
        pbar_size = samples_num
        pbar_size += metrics_num
        if self.optimizer is not None:
            pbar_size += round(self.optimizer.n_trials * samples_num * (1 - test_size))
            pbar_size += samples_num
            pbar_size += metrics_num
        self.pbar.init_bar(pbar_size)

    def train(self, X, y):
        embeded_x = self.embedding.transform(X)
        self.regressor.fit(embeded_x, y)

    def test(self, X):
        embeded_x = self.embedding.transform(X)
        preds = self.regressor.predict(embeded_x)
        return preds

    @abstractmethod
    def train_test_split(self, x, y, test_size, shuffle, random_state):
        pass

    def evaluate(self, X_train, X_test, y_train, y_test):
        x_emb_train = numpy.array(self.embedding.transform(X_train))
        x_emb_test = numpy.array(self.embedding.transform(X_test))
        y_train = numpy.array(y_train)
        y_test = numpy.array(y_test)
        self.regressor.fit(x_emb_train, y_train)
        y_pred = self.regressor.predict(x_emb_test)
        return y_pred, y_test

    def set_params(self, params: dict):
        emb_space = [par['name'] for par in self.embedding.space()]
        regr_space = [par['name'] for par in self.regressor.space()]
        emb_params = dict()
        for key, val in params.items():
            if key not in emb_space:
                continue
            emb_params[key] = val
        self.embedding.set_params(emb_params)
        regr_params = dict()
        for key, val in params.items():
            if key not in regr_space:
                continue
            regr_params[key] = val
        self.regressor.set_params(regr_params)

    def score(self, X: list, y: list,
              report: Report,
              test_size=0.25,
              valid_size=0.25,
              random_state=42,
              shuffle=True):
        self.init_pbar(len(X), test_size, report)
        self.pbar.set_predescription(self.name)
        self.pbar.set_description('Default model params...')
        X_train, X_test, y_train, y_test = \
            self.train_test_split(X, y, test_size=test_size, shuffle=shuffle, random_state=random_state)
        y_pred, y_test = self.evaluate(X_train, X_test, y_train, y_test)
        self.pbar.set_description('Scoring...')
        report.score(y_pred, y_test, self.name)
        if self.optimizer is None:
            self.pbar.set_description('Done')
            return report
        self.pbar.set_description('Model params optimizing...')
        self.name += '_optim'
        X_train, X_test, y_train, y_test =\
            self.train_test_split(X, y, test_size=test_size, shuffle=shuffle, random_state=random_state)
        X_train_optim, X_valid_optim, y_train_optim, y_valid_optim =\
            self.train_test_split(X_train, y_train, test_size=valid_size, shuffle=shuffle, random_state=random_state)
        try:
            best_params = self.optimizer.optimize(X_train_optim, X_valid_optim, y_train_optim, y_valid_optim,
                                                  self.embedding,
                                                  regressor=self.regressor)
            self.set_params(best_params)
            y_pred, y_test = self.evaluate(X_train, X_test, y_train, y_test)
        except ValueError:
            y_pred = None
            self.pbar.add(len(X))
        report.score(y_pred, y_test, self.name)

        self.pbar.finish()
        return report
