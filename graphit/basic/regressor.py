from . import Regressor

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from lightgbm import LGBMRegressor


class PCARegressor(Regressor):
    def __init__(self, **params):
        """
        :param params: dict of
            n_components: int, float or ‘mle’, default=None
            tol: float, default=0.0
            copy: bool, default=True
            whiten: bool, default=False
            svd_solver: {‘auto’, ‘full’, ‘arpack’, ‘randomized’}, default=’auto’
            iterated_power: int or ‘auto’, default=’auto’
            random_state: int, RandomState instance or None, default=None
        """
        if 'n_components' not in params.keys():
            params['n_components'] = 1
        self.model = PCA(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, x):
        return self.model.transform(x).squeeze()

    def space(self):
        params = [
            {'name': 'whiten',
             'type': 'bool'},
            {'name': 'svd_solver',
             'type': 'list',
             'values': ['auto', 'full', 'arpack', 'randomized']},
            {'name': 'iterated_power',
             'type': 'int',
             'from': 1,
             'to': 100}
        ]
        return params

    def set_params(self, params: dict):
        for name, val in params.items():
            setattr(self.model, name, val)


class LinearRegressor(Regressor):
    def __init__(self, **params):
        """
        :param params: dict of
            n_components: int, float or ‘mle’, default=None
            tol: float, default=0.0
            copy: bool, default=True
            whiten: bool, default=False
            svd_solver: {‘auto’, ‘full’, ‘arpack’, ‘randomized’}, default=’auto’
            iterated_power: int or ‘auto’, default=’auto’
            random_state: int, RandomState instance or None, default=None
        """
        self.model = LinearRegression(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, x):
        return self.model.predict(x)

    def space(self):
        params = [
            {'name': 'n_jobs',
             'type': 'int',
             'from': 4,
             'to': 4},
        ]
        return params

    def set_params(self, params: dict):
        for name, val in params.items():
            setattr(self.model, name, val)


class LGBM(Regressor):
    def __init__(self, **params):
        self.model = LGBMRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, x):
        return self.model.predict(x)

    def space(self):
        params = [
            {'name': 'num_leaves ',
             'type': 'int',
             'from': 5,
             'to': 100},
            {'name': 'max_depth',
             'type': 'int',
             'from': -1,
             'to': 1000},
            {'name': 'n_estimators',
             'type': 'int',
             'from': 10,
             'to': 1000},
            {'name': 'subsample_for_bin',
             'type': 'int',
             'from': 50000,
             'to': 1000000},
            {'name': 'subsample_freq',
             'type': 'int',
             'from': 0,
             'to': 1000},
            {'name': 'boosting_type',
             'type': 'list',
             'values': ['gbdt', 'dart', 'goss']},
            {'name': 'learning_rate',
             'type': 'float',
             'from': 0.001,
             'to': 0.1},
            {'name': 'min_split_gain',
             'type': 'float',
             'from': 0.0,
             'to': 0.5},
            {'name': 'min_child_weight',
             'type': 'float',
             'from': 0.0001,
             'to': 0.1}
        ]
        return params


    def set_params(self, params: dict):
        for name, val in params.items():
            setattr(self.model, name, val)
