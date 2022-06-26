from abc import ABC, abstractmethod


class ProgressBar(ABC):
    def __init__(self):
        self.bar_size = 0
        self.predescription = None

    def init_bar(self, size: int):
        self.bar_size = size

    def next(self):
        self.add(1)

    def set_predescription(self, predescription):
        self.predescription = predescription

    def predescribe(self, text):
        text = self.predescription + ' | ' + text
        return text

    @abstractmethod
    def add(self, n):
        pass

    @abstractmethod
    def set_description(self, text: str):
        pass

    def finish(self):
        self.set_description('Done')
