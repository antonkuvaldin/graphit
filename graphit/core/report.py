from abc import ABC, abstractmethod
from typing import Callable, Optional
from .core import Process

import math
import numpy
import pingouin


class Metric(ABC):
    def __init__(self, name: str, function: Callable):
        self._name = name
        self._func = function
        self._value = None

    def calc(self, *args) -> float:
        try:
            self._value = self._func(*args)
        except Exception as e:
            # TODO: Внедрить модуль logging
            print(f"Can't calculate {self.name}!\n{e}")
            return math.nan
        return self._value

    @property
    def name(self):
        return self._name

    @property
    def func(self):
        return self._func

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_val):
        self._value = new_val

    def is_calculated(self) -> bool:
        if self.value is None:
            return False
        return True


class Report(Process):
    def __init__(self, metrics: dict, n_boots=10000, sample_size=None, alpha=0.95, seed=42):
        #TODO добавить параметр 'рамер семпла на каждой итерации'
        self._metrics = self.parse_dict(metrics)
        self.alpha = alpha
        self._results = dict()
        self.n_boots = n_boots
        self.sample_size = sample_size
        self.seed = seed

    def parse_dict(self, metrics: dict):
        result_metrics = [Metric(m_name, m_func) for m_name, m_func in metrics.items()]
        return result_metrics

    @abstractmethod
    def calc_conf_interval(self, distr, alpha: float):
        pass

    def get_bootstrap_idxs(self, dist_size, n_iter, bts_size):
        bts_num = round(dist_size * bts_size)
        all_idxs = numpy.arange(dist_size)
        samples = []
        for _ in range(n_iter):
            samples.append(list(numpy.random.choice(all_idxs, size=bts_num, replace=True)))
        return samples

    #TODO: Избавиться от зависимостей: библиотека pingouin
    def calc_metrics(self, predictions: list, target: list, model: str, label: str, metric_type: str):
        self.pbar.set_description('Scoring...')
        current_results = dict()
        predictions = numpy.array(predictions) if predictions is not None else predictions
        target = numpy.array(target)
        for metric in self._metrics:
            metric_name = metric.name
            if label is not None:
                metric_name += f'_{label}'
            if predictions is None:
                current_results[metric_name] = {'value': None, 'confidence': None}
                self.pbar.next()
                continue
            metric_res = metric.calc(target, predictions)
            def f(*args):
                return metric.calc(*args)
            metric_confidence = pingouin.compute_bootci(target,
                                                        predictions,
                                                        func=f,
                                                        seed=self.seed,
                                                        n_boot=self.n_boots,
                                                        confidence=self.alpha)
            current_results[metric_name] = {'value': round(metric_res, 2), 'confidence': f"{metric_confidence}"}
            self.pbar.next()
        self.log_metric_result(current_results, metric_type, model)

    def log_metric_result(self, result, metric_type, model):
        if model in self._results.keys():
            new_result = {**self._results[model], **result}
        else:
            new_result = result
        self._results[model] = new_result

    def score(self, predictions: list, target: list, model: str, label: Optional[str]=None) -> None:
        self.calc_metrics(predictions, target, model, label, metric_type='score')

    @abstractmethod
    def show(self):
        pass
