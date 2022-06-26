from . import Optimizer

import copy
import optuna
import sklearn


class OptunaOptimizer(Optimizer):
    def __init__(self, n_jobs=-1, **kwargs):
        self.n_jobs = n_jobs
        super().__init__(**kwargs)

    def parse_params(self, space, **kwargs):
        trial = kwargs['trial']
        parsed_params = dict()
        for param in space:
            if param['type'] == 'int':
                parsed_params[param['name']] = trial.suggest_int(param['name'], param['from'], param['to'])
            elif param['type'] == 'float':
                parsed_params[param['name']] = trial.suggest_float(param['name'], param['from'], param['to'])
            elif param['type'] == 'bool':
                parsed_params[param['name']] = trial.suggest_categorical(param['name'], [True, False])
            elif param['type'] == 'list':
                parsed_params[param['name']] = trial.suggest_categorical(param['name'], param['values'])
            else:
                raise ValueError(f"Unsupported type of param: {param['type']}!")
        return parsed_params

    def optimize(self, X_train, X_test, y_train, y_test, embedding, regressor=None):
        emb_space = embedding.space()
        if regressor is not None:
            regr_space = regressor.space()
        else:
            regr_space = None
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(direction=self.direction)
        def objective(trial):
            emb_params = self.parse_params(emb_space, trial=trial)
            curr_emb_model = copy.deepcopy(embedding)
            curr_emb_model.set_params(emb_params)
            if regr_space is not None:
                regr_params = self.parse_params(regr_space, trial=trial)
                curr_regression_model = copy.deepcopy(regressor)
                curr_regression_model.set_params(regr_params)
                X_train_emb = embedding.transform(X_train)
                curr_regression_model.fit(X_train_emb, y_train)
                X_test_emb = embedding.transform(X_test)
                predicted = curr_regression_model.predict(X_test_emb)
            else:
                predicted = embedding.transform(X_train)
            result = self.optim_func(y_test, predicted)
            return result
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False, n_jobs=self.n_jobs)
        return self.study.best_params
