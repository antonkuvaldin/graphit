from abc import ABC, abstractmethod
from .core import Estimator, Tunable, Process


class Regressor(Estimator, Tunable, Process):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
