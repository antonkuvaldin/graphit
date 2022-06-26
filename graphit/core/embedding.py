from abc import abstractmethod
from .core import Estimator, Process, Tunable
from collections.abc import Collection

import numpy


class Embedding(Estimator, Tunable, Process):
    def __init__(self, **kwargs):
        self.threshold = kwargs.pop('th', None)

    def transform(self, samples: Collection) -> numpy.array:
        transformed_samples = list()
        for smpl in samples:
            transformed_samples.append(self.get_embedding(smpl))
        return numpy.array(transformed_samples)

    def get_embedding(self, sample):
        sample = numpy.array(sample)
        if self.threshold is not None:
            sample[numpy.abs(sample) < self.threshold] = 0
        self.pbar.next()
        return self.embedding(sample)

    @abstractmethod
    def embedding(self, sample):
        pass

    def embedding_to_number(self, embedding):
        embedding = numpy.array(embedding)
        if embedding.ndim > 1:
            embedding = embedding.mean(axis=tuple(range(1, embedding.ndim)))
        return embedding
