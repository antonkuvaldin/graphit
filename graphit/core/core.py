from abc import ABC, abstractmethod
import copy
import pickle

from . import ProgressBar

from IPython import get_ipython
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


class Estimator(ABC):
    @abstractmethod
    def set_params(self, params: dict):
        pass


class Tunable(ABC):
    def space(self) -> list:
        return list()


class Process():
    def set_pbar(self, pbar: ProgressBar):
        self.pbar = pbar

    def __deepcopy__(self, memo):
        cls = self.__class__ # Extract the class of the object
        result = cls.__new__(cls) # Create a new instance of the object based on extracted class
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            new_v = v
            if k != 'pbar':
                new_v = copy.deepcopy(v, memo)
            setattr(result, k, new_v)
        return result

    def to_pickle(self, path: str, **params):
        self.set_pbar(None)
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_pickle(path: str):
        with open(path, 'rb') as handle:
            obj = pickle.load(handle)
        return obj

