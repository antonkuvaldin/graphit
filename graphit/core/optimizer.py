from abc import ABC, abstractmethod
from .core import Process


class Optimizer(Process):
    def __init__(self, n_trials=10, direction='minimize', optim_func=None):
        self.n_trials = n_trials
        self.direction = direction
        self.optim_func = optim_func

    @abstractmethod
    def parse_params(self, space, **kwargs):
        pass

    @abstractmethod
    def optimize(self, X_train, X_test, y_train, y_test, embedding, regressor=None):
        pass
