from . import Embedding

import numpy
import networkx
import karateclub


class BasicEmbedding(Embedding):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def embedding(self, x):
        x = numpy.array(x)
        embedding = numpy.array(numpy.mean(x, axis=1))
        return embedding

    def set_model(self, model):
        self.model = model

    def set_params(self, params: dict):
        if self.model is None:
            return
        for name, val in params.items():
            setattr(self.model, name, val)

    def space(self):
        params = [
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.9}
        ]
        return params


class RandomEmbedding(BasicEmbedding):
    def __init__(self, size=10, **kwargs):
        self.size = size
        super().__init__(**kwargs)

    def embedding(self, x):
        return numpy.random.rand(self.size)

    def space(self):
        params = [
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.9}
        ]
        return params


class CentralityEmbedding(BasicEmbedding):
    def __init__(self, **params):
        self.func = params.pop('func', None)
        super().__init__()

    def embedding(self, x):
        G = networkx.from_numpy_matrix(x, create_using=networkx.Graph)
        embedding = numpy.array(list(self.func(G).values()))
        return embedding


class KarateclubEmbedding(BasicEmbedding):
    def embedding(self, x):
        # TODO: добавить выбор ориентированного/не ориентированного графа
        # TODO: добавить алгоритмы, принимающие фичи вершин в качестве аргумента в методе fit
        G = networkx.from_numpy_matrix(x, create_using=networkx.DiGraph)
        self.model.fit(G)
        embedding = self.model.get_embedding()
        if embedding.ndim == 2:
            embedding = numpy.mean(embedding, axis=1)
        return embedding


class KarateclubListGraphEmbedding(KarateclubEmbedding):
    def transform(self, X, mean=False):
        graphs = list()
        for sample in X:
            graphs.append(networkx.from_numpy_matrix(sample, create_using=networkx.DiGraph))
        self.model.fit(graphs)
        self.pbar.add(len(graphs))
        transformed_samples = self.model.get_embedding()
        if mean:
            transformed_samples = transformed_samples.mean(axis=tuple(range(1, transformed_samples.ndim)))
        return transformed_samples


class Graph2VecEmbedding(KarateclubListGraphEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            wl_iterations (int) – Number of Weisfeiler-Lehman iterations. Default is 2.
            attributed (bool) – Presence of graph attributes. Default is False.
            dimensions (int) – Dimensionality of embedding. Default is 128.
            workers (int) – Number of cores. Default is 4.
            down_sampling (float) – Down sampling frequency. Default is 0.0001.
            epochs (int) – Number of epochs. Default is 10.
            learning_rate (float) – HogWild! learning rate. Default is 0.025.
            min_count (int) – Minimal count of graph feature occurrences. Default is 5.
            seed (int) – Random seed for the model. Default is 42.
            erase_base_features (bool) – Erasing the base features. Default is False.
        """
        super().__init__()
        self.set_model(karateclub.Graph2Vec(**params))

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 300},
            {'name': 'wl_iterations',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'down_sampling',
             'type': 'float',
             'from': 0.00001,
             'to': 0.001},
            {'name': 'epochs',
             'type': 'int',
             'from': 3,
             'to': 20},
            {'name': 'learning_rate',
             'type': 'float',
             'from': 0.001,
             'to': 0.1},
            {'name': 'min_count',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class FGSDEmbedding(KarateclubListGraphEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            hist_bins (int) – Number of histogram bins. Default is 200.
            hist_range (int) – Histogram range considered. Default is 20.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.FGSD(**params))

    def transform(self, X, mean=False):
        graphs = list()
        for sample in X:
            graphs.append(networkx.from_numpy_matrix(sample, create_using=networkx.DiGraph).to_undirected())
        self.model.fit(graphs)
        self.pbar.add(len(graphs))
        transformed_samples = self.model.get_embedding()
        if mean:
            transformed_samples = transformed_samples.mean(axis=tuple(range(1, transformed_samples.ndim)))
        return transformed_samples

    def space(self):
        params = [
            {'name': 'hist_bins',
             'type': 'int',
             'from': 10,
             'to': 500},
            {'name': 'hist_range',
             'type': 'int',
             'from': 1,
             'to': 50},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class NetLSDEmbedding(KarateclubListGraphEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            scale_min (float) – Time scale interval minimum. Default is -2.0.
            scale_max (float) – Time scale interval maximum. Default is 2.0.
            scale_steps (int) – Number of steps in time scale. Default is 250.
            scale_approximations (int) – Number of eigenvalue approximations. Default is 200.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.NetLSD(**params))

    def transform(self, X, mean=False):
        graphs = list()
        for sample in X:
            graphs.append(networkx.from_numpy_matrix(sample, create_using=networkx.DiGraph).to_undirected())
        self.model.fit(graphs)
        self.pbar.add(len(graphs))
        transformed_samples = self.model.get_embedding()
        if mean:
            transformed_samples = transformed_samples.mean(axis=tuple(range(1, transformed_samples.ndim)))
        return transformed_samples

    def space(self):
        params = [
            {'name': 'scale_steps',
             'type': 'int',
             'from': 10,
             'to': 500},
            {'name': 'scale_min',
             'type': 'float',
             'from': -10.0,
             'to': -0.001},
            {'name': 'scale_max',
             'type': 'float',
             'from': 0.001,
             'to': 10.0},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class GeoScatteringEmbedding(KarateclubListGraphEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            order (int) – Adjacency matrix powers. Default is 4.
            moments (int) – Unnormalized moments considered. Default is 4.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.GeoScattering(**params))

    def space(self):
        params = [
            {'name': 'order',
             'type': 'int',
             'from': 1,
             'to': 8},
            {'name': 'moments',
             'type': 'int',
             'from': 1,
             'to': 8},
            {'name': 'th',
             'type': 'float',
             'from': 0.0,
             'to': 0.0}
        ]
        return params


class FeatherGraphEmbedding(KarateclubListGraphEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            order (int) – Adjacency matrix powers. Default is 5.
            eval_points (int) – Number of evaluation points. Default is 25.
            theta_max (int) – Maximal evaluation point value. Default is 2.5.
            seed (int) – Random seed value. Default is 42.
            pooling (str) – Permutation invariant pooling function, one of: ("mean", "max", "min"). Default is “mean.”
        """
        super().__init__()
        self.set_model(karateclub.FeatherGraph(**params))

    def space(self):
        params = [
            {'name': 'order',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'eval_points',
             'type': 'int',
             'from': 1,
             'to': 50},
            {'name': 'theta_max',
             'type': 'float',
             'from': 0.1,
             'to': 5.0},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class LDPEmbedding(KarateclubListGraphEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            bins (int) – Number of histogram bins. Default is 32.
        """
        super().__init__()
        self.set_model(karateclub.LDP(**params))

    def space(self):
        params = [
            {'name': 'bins',
             'type': 'int',
             'from': 1,
             'to': 100},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class WaveletCharacteristicEmbedding(KarateclubListGraphEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            order (int) – Adjacency matrix powers. Default is 5.
            eval_points (int) – Number of characteristic function evaluations. Default is 5.
            theta_max (float) – Largest characteristic function time value. Default is 2.5.
            tau (float) – Wave function heat - time diffusion. Default is 1.0.
            pooling (str) – Pooling function appliead to the characteristic functions. Default is “mean”.
        """
        super().__init__()
        self.set_model(karateclub.WaveletCharacteristic(**params))

    def transform(self, X, mean=False):
        graphs = list()
        for sample in X:
            graphs.append(networkx.from_numpy_matrix(sample, create_using=networkx.DiGraph).to_undirected())
        self.model.fit(graphs)
        self.pbar.add(len(graphs))
        transformed_samples = self.model.get_embedding()
        if mean:
            transformed_samples = transformed_samples.mean(axis=tuple(range(1, transformed_samples.ndim)))
        return transformed_samples

    def space(self):
        params = [
            {'name': 'order',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'eval_points',
             'type': 'int',
             'from': 1,
             'to': 50},
            {'name': 'theta_max',
             'type': 'float',
             'from': 0.1,
             'to': 5.0},
            {'name': 'tau',
             'type': 'float',
             'from': 0.1,
             'to': 2.0},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params

class DeepWalkEmbedding(KarateclubEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            walk_number (int) – Number of random walks. Default is 10.
            walk_length (int) – Length of random walks. Default is 80.
            dimensions (int) – Dimensionality of embedding. Default is 128.
            workers (int) – Number of cores. Default is 4.
            window_size (int) – Matrix power order. Default is 5.
            epochs (int) – Number of epochs. Default is 1.
            learning_rate (float) – HogWild! learning rate. Default is 0.05.
            min_count (int) – Minimal count of node occurrences. Default is 1.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.DeepWalk(**params))

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 300},
            {'name': 'walk_number',
             'type': 'int',
             'from': 3,
             'to': 50},
            {'name': 'walk_length',
             'type': 'int',
             'from': 10,
             'to': 100},
            {'name': 'window_size',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'epochs',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'learning_rate',
             'type': 'float',
             'from': 0.001,
             'to': 0.1},
            {'name': 'min_count',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class Role2VecEmbedding(KarateclubEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            walk_number (int) – Number of random walks. Default is 10.
            walk_length (int) – Length of random walks. Default is 80.
            dimensions (int) – Dimensionality of embedding. Default is 128.
            workers (int) – Number of cores. Default is 4.
            window_size (int) – Matrix power order. Default is 2.
            epochs (int) – Number of epochs. Default is 1.
            learning_rate (float) – HogWild! learning rate. Default is 0.05.
            down_sampling (float) – Down sampling frequency. Default is 0.0001.
            min_count (int) – Minimal count of feature occurrences. Default is 10.
            wl_iterations (int) – Number of Weisfeiler-Lehman hashing iterations. Default is 2.
            seed (int) – Random seed value. Default is 42.
            erase_base_features (bool) – Removing the base features. Default is False.
        """
        super().__init__()
        self.set_model(karateclub.Role2Vec(**params))

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 300},
            {'name': 'walk_number',
             'type': 'int',
             'from': 3,
             'to': 50},
            {'name': 'walk_length',
             'type': 'int',
             'from': 10,
             'to': 100},
            {'name': 'window_size',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'epochs',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'learning_rate',
             'type': 'float',
             'from': 0.001,
             'to': 0.1},
            {'name': 'down_sampling',
             'type': 'float',
             'from': 0.00001,
             'to': 0.01},
            {'name': 'min_count',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'wl_iterations',
             'type': 'int',
             'from': 1,
             'to': 5},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class Node2VecEmbedding(KarateclubEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            walk_number (int) – Number of random walks. Default is 10.
            walk_length (int) – Length of random walks. Default is 80.
            p (float) – Return parameter (1/p transition probability) to move towards from previous node.
            q (float) – In-out parameter (1/q transition probability) to move away from previous node.
            dimensions (int) – Dimensionality of embedding. Default is 128.
            workers (int) – Number of cores. Default is 4.
            window_size (int) – Matrix power order. Default is 5.
            epochs (int) – Number of epochs. Default is 1.
            learning_rate (float) – HogWild! learning rate. Default is 0.05.
            min_count (int) – Minimal count of node occurrences. Default is 1.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.Node2Vec(**params))

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 300},
            {'name': 'walk_number',
             'type': 'int',
             'from': 3,
             'to': 50},
            {'name': 'walk_length',
             'type': 'int',
             'from': 10,
             'to': 100},
            {'name': 'window_size',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'epochs',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'learning_rate',
             'type': 'float',
             'from': 0.001,
             'to': 0.1},
            {'name': 'p',
             'type': 'float',
             'from': 0.0,
             'to': 1.0},
            {'name': 'q',
             'type': 'float',
             'from': 0.0,
             'to': 1.0},
            {'name': 'min_count',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class LaplacianEigenmapsEmbedding(KarateclubEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            dimensions (int) – Dimensionality of embedding. Default is 128.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.LaplacianEigenmaps(**params))

    def embedding(self, x):
        # TODO: добавить выбор ориентированного/не ориентированного графа
        # TODO: добавить алгоритмы, принимающие фичи вершин в качестве аргумента в методе fit
        G = networkx.from_numpy_matrix(x, create_using=networkx.DiGraph).to_undirected()
        self.model.fit(G)
        embedding = self.model.get_embedding()
        if embedding.ndim == 2:
            embedding = numpy.mean(embedding, axis=1)
        return embedding

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 50},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class GraRepEmbedding(KarateclubEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            dimensions (int) – Number of individual embedding dimensions. Default is 32.
            iteration (int) – Number of SVD iterations. Default is 10.
            order (int) – Number of PMI matrix powers. Default is 5.
            seed (int) – SVD random seed. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.GraRep(**params))

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 50},
            {'name': 'iteration',
             'type': 'int',
             'from': 1,
             'to': 20},
            {'name': 'order',
             'type': 'int',
             'from': 3,
             'to': 7},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class BoostNEEmbedding(KarateclubEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            dimensions (int) – Number of individual embedding dimensions. Default is 8.
            iterations (int) – Number of boosting iterations. Default is 16.
            order (int) – Number of adjacency matrix powers. Default is 2.
            alpha (float) – NMF regularization parameter. Default is 0.01.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.BoostNE(**params))

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 300},
            {'name': 'iteration',
             'type': 'int',
             'from': 1,
             'to': 20},
            {'name': 'order',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'alpha',
             'type': 'float',
             'from': 0.001,
             'to': 0.1},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class NodeSketchEmbedding(KarateclubEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            dimensions (int) – Embedding dimensions. Default is 32.
            iterations (int) – Number of iterations (sketch order minus one). Default is 2.
            decay (float) – Exponential decay rate. Default is 0.01.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.NodeSketch(**params))

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 300},
            {'name': 'iteration',
             'type': 'int',
             'from': 1,
             'to': 20},
            {'name': 'decay',
             'type': 'float',
             'from': 0.001,
             'to': 0.1},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class WalkletsEmbedding(KarateclubEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            walk_number (int) – Number of random walks. Default is 10.
            walk_length (int) – Length of random walks. Default is 80.
            dimensions (int) – Dimensionality of embedding. Default is 32.
            workers (int) – Number of cores. Default is 4.
            window_size (int) – Matrix power order. Default is 4.
            epochs (int) – Number of epochs. Default is 1.
            learning_rate (float) – HogWild! learning rate. Default is 0.05.
            min_count (int) – Minimal count of node occurrences. Default is 1.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.Walklets(**params))

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 300},
            {'name': 'walk_number',
             'type': 'int',
             'from': 3,
             'to': 50},
            {'name': 'walk_length',
             'type': 'int',
             'from': 10,
             'to': 100},
            {'name': 'window_size',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'epochs',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'learning_rate',
             'type': 'float',
             'from': 0.001,
             'to': 0.1},
            {'name': 'min_count',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class GL2VecEmbedding(KarateclubListGraphEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            wl_iterations (int) – Number of Weisfeiler-Lehman iterations. Default is 2.
            workers (int) – Number of cores. Default is 4.
            dimensions (int) – Dimensionality of embedding. Default is 128.
            down_sampling (float) – Down sampling frequency. Default is 0.0001.
            epochs (int) – Number of epochs. Default is 10.
            learning_rate (float) – HogWild! learning rate. Default is 0.05.
            min_count (int) – Minimal count of node occurrences. Default is 1.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.GL2Vec(**params))

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 300},
            {'name': 'wl_iterations',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'down_sampling',
             'type': 'float',
             'from': 0.00001,
             'to': 0.001},
            {'name': 'epochs',
             'type': 'int',
             'from': 3,
             'to': 20},
            {'name': 'learning_rate',
             'type': 'float',
             'from': 0.001,
             'to': 0.1},
            {'name': 'min_count',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class SFEmbedding(KarateclubListGraphEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            dimensions (int) – Dimensionality of embedding. Default is 128.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.SF(**params))

    def transform(self, X, mean=False):
        graphs = list()
        for sample in X:
            graphs.append(networkx.from_numpy_matrix(sample, create_using=networkx.DiGraph).to_undirected())
        self.model.fit(graphs)
        self.pbar.add(len(graphs))
        transformed_samples = self.model.get_embedding()
        if mean:
            transformed_samples = transformed_samples.mean(axis=tuple(range(1, transformed_samples.ndim)))
        return transformed_samples

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 300},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class GLEEEmbedding(KarateclubEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            dimensions (int) – Dimensionality of embedding. Default is 128.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.GLEE(**params))

    def embedding(self, x):
        # TODO: добавить выбор ориентированного/не ориентированного графа
        G = networkx.from_numpy_matrix(x, create_using=networkx.DiGraph).to_undirected()
        self.model.fit(G)
        embedding = self.model.get_embedding()
        if embedding.ndim == 2:
            embedding = numpy.mean(embedding, axis=1)
        return embedding

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 300},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class SocioDimEmbedding(KarateclubEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            dimensions (int) – Dimensionality of embedding. Default is 128.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.SocioDim(**params))

    def embedding(self, x, mean=False):
        G = networkx.from_numpy_matrix(x, create_using=networkx.Graph)
        self.model.fit(G)
        embedding = self.model.get_embedding()
        #print(f"BEFORE: {embedding.shape}")
        #embedding = numpy.mean(embedding, axis=0)
        if not mean:
            embedding = embedding.ravel()
        elif embedding.ndim != 1:
            embedding = numpy.mean(embedding)
        #print(f"AFTER: {embedding.shape}")
        return embedding

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 300},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class GEMSECEmbedding(KarateclubEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            dimensions (int) – Dimensionality of embedding. Default is 128.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.GEMSEC(**params))

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 300},
            {'name': 'walk_number',
             'type': 'int',
             'from': 3,
             'to': 50},
            {'name': 'walk_length',
             'type': 'int',
             'from': 10,
             'to': 100},
            {'name': 'window_size',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'learning_rate',
             'type': 'float',
             'from': 0.001,
             'to': 0.1},
            {'name': 'negative_samples',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'gamma',
             'type': 'float',
             'from': 0.01,
             'to': 0.9},
            {'name': 'clusters',
             'type': 'int',
             'from': 1,
             'to': 20},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class NNSEDEmbedding(KarateclubEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            dimensions (int) – Dimensionality of embedding. Default is 128.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.NNSED(**params))

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 100},
            {'name': 'iterations',
             'type': 'int',
             'from': 1,
             'to': 20},
            {'name': 'noise',
             'type': 'float',
             'from': 1e-07,
             'to': 1e-05},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class BigClamEmbedding(KarateclubEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            dimensions (int) – Dimensionality of embedding. Default is 128.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.BigClam(**params))

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 100},
            {'name': 'iterations',
             'type': 'int',
             'from': 1,
             'to': 20},
            {'name': 'learning_rate',
             'type': 'float',
             'from': 0.0005,
             'to': 0.05},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params


class SymmNMFEmbedding(KarateclubEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            dimensions (int) – Number of dimensions. Default is 32.
            iterations (int) – Number of power iterations. Default is 200.
            rho (float) – Regularization tuning parameter. Default is 100.0.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.SymmNMF(**params))

    def space(self):
        params = [
            {'name': 'dimensions',
             'type': 'int',
             'from': 30,
             'to': 34},
            {'name': 'iterations',
             'type': 'int',
             'from': 198,
             'to': 202},
            {'name': 'rho',
             'type': 'float',
             'from': 98.0,
             'to': 102.0},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.1}
        ]
        return params


class Diff2VecEmbedding(KarateclubEmbedding):
    def __init__(self, **params):
        """
        :param params: dict of
            wl_iterations (int) – Number of Weisfeiler-Lehman iterations. Default is 2.
            workers (int) – Number of cores. Default is 4.
            dimensions (int) – Dimensionality of embedding. Default is 128.
            down_sampling (float) – Down sampling frequency. Default is 0.0001.
            epochs (int) – Number of epochs. Default is 10.
            learning_rate (float) – HogWild! learning rate. Default is 0.05.
            min_count (int) – Minimal count of node occurrences. Default is 1.
            seed (int) – Random seed value. Default is 42.
        """
        super().__init__()
        self.set_model(karateclub.Diff2Vec(**params))

    def space(self):
        params = [
            {'name': 'diffusion_number',
             'type': 'int',
             'from': 1,
             'to': 20},
            {'name': 'diffusion_cover',
             'type': 'int',
             'from': 50,
             'to': 150},
            {'name': 'dimensions',
             'type': 'int',
             'from': 1,
             'to': 300},
            {'name': 'window_size',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'epochs',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'learning_rate',
             'type': 'float',
             'from': 0.005,
             'to': 0.5},
            {'name': 'min_count',
             'type': 'int',
             'from': 1,
             'to': 10},
            {'name': 'th',
             'type': 'float',
             'from': 0.01,
             'to': 0.3}
        ]
        return params
