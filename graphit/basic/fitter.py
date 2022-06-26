from . import Fitter

from .progress_bar import TQDMProgressBar
import sklearn

class BasicFitter(Fitter):
    def __init__(self, *args, **kwargs):
        super(BasicFitter, self).__init__(*args, **kwargs)
        if self.pbar is None:
            self.pbar = TQDMProgressBar()

    def train_test_split(self, x, y, test_size, shuffle, random_state):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y,
                                                                                    test_size=test_size,
                                                                                    shuffle=shuffle,
                                                                                    random_state=random_state)
        return X_train, X_test, y_train, y_test
