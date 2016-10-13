from __future__ import print_function

import numpy as np
import pandas as pd
import pickle
import copy
from contextlib import contextmanager
import random
import six

import src.utils as utils

def from_config(config):
    tracks = Tracks(config.get('tracks', []))
    fixed = None
    if 'fixed' in config:
        fixed = Fixed(config['fixed'])
    if ('labels' in config) and ('regions' in config):
        raise Exception('Cannot specify both labels and regions in config.')
    elif 'labels' in config:
        regions = Regions(config['labels'], key='labels')
    elif 'regions' in config:
        regions = Regions(config['regions'], key='regions')
    seed = random.getrandbits(32)
    if 'seed' in config:
        seed = config['seed']
    data = src.loader.RegionsIterator(tracks, fixed, regions, seed=seed)
    return data

def regions(path):
    if path.endswith('.gz'):
        import gzip
        with gzip.open(path) as f:
            array = pickle.load(f).values
    else:
        with open(path, 'rb') as f:
            array = pickle.load(f).values
    if array.shape[1] > 4:
        return Labels(path, array)
    return Regions(path, array)

class Regions(object):
    def __init__(self, path, array):
        self.path = path
        self.regions = array

    @property
    def key(self):
        return 'regions'

    def to_config(self):
        cell_lines = self.cell_lines()
        config = {'path': self.path}
        config['cell_lines'] = sorted(list(cell_lines))
        return config

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, i):
        if hasattr(i, '__len__'):
            cpy = copy.copy(self)
            cpy.regions = self.regions[i]
            return cpy
        return self.regions[i]

    def __iter__(self):
        return iter(self.regions)

    def cell_lines(self):
        return set(self.regions[:,0])

class Labels(object):
    def __init__(self, path, array):
        self.path = path
        self.Y = self.to_matrix(array[:,4])
        self.Y_M = self.to_matrix(array[:,5])
        self.regions = array[:,:4]

    def to_matrix(self, v):
        m = len(v)
        n = max(len(x) for x in v)
        A = np.zeros((m,n), dtype=v[0].dtype)
        for i in range(m):
            k = len(v[i])
            A[i,:k] = v[i]
        return A

    @property
    def key(self):
        return 'labels'

    def to_config(self):
        cell_lines = self.cell_lines()
        config = {'path': self.path}
        config['cell_lines'] = sorted(list(cell_lines))
        return config

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, i):
        if hasattr(i, '__len__'):
            cpy = copy.copy(self)
            cpy.Y = self.Y[i]
            cpy.Y_M = self.Y_M[i]
            cpy.regions = self.regions[i]
            return cpy
        return self.regions[i], self.Y[i], self.Y_M[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self.regions[i], self.Y[i], self.Y_M[i]

    def cell_lines(self):
        return set(self.regions[:,0])


def find_file(paths, cell_line):
    for path in paths:
        if cell_line in path:
            return path

class Tracks(object):
    def __init__(self, paths):
        self.paths = paths

    def to_config(self):
        return self.paths

    @contextmanager
    def open(self, cell_lines):
        di = {cell_line: find_file(self.paths, cell_line) for cell_line in cell_lines}
        loader = HDF5Loader(di)
        try:
            yield loader
        finally:
            loader.close()

class HDF5Loader(object):
    def __init__(self, cell_lines):
        self.stores = {cell_line:pd.HDFStore(path, mode='r') for cell_line,path in six.iteritems(cell_lines)}

    def __getitem__(self, i):
        cell_line,chrom,start,end = i[0], i[1], i[2], i[3]
        store = self.stores[cell_line]
        return store.select(chrom, 'index>=start & index<end').values       

    def close(self):
        for k,v in self.stores.items():
            v.close()

class Fixed(object):
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path, sep='\t', index_col=0)

    def to_config(self):
        return self.path

    def __getitem__(self, i):
        series = self.df[i]
        return series.values

@contextmanager
def features(tracks, fixed, cell_lines):
    with tracks.open(cell_lines) as tr:
        yield Features(tr, fixed)

class Features(object):
    def __init__(self, tracks, fixed):
        self.tracks = tracks
        self.fixed = fixed

    def __getitem__(self, i):
        X = self.tracks[i]
        cell_line = i[0]
        if self.fixed is not None:
            F = self.fixed[cell_line]
        else:
            F = None
        return X, F

class RegionsIterator(object):
    def __init__(self, tracks, fixed, regions, randomize=False, infinite=False, weights=None, chunk_size=1, seed=None, dtype=np.float32):
        self.tracks = tracks
        self.fixed = fixed
        self.regions = regions
        self.n = max_region(regions.regions)
        self.randomize = randomize
        self.infinite = infinite
        self.chunk_size = chunk_size
        self.weights = weights
        if seed is None:
            seed = random.getrandbits(32)
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.dtype = dtype

    def to_config(self):
        config = {'tracks': self.tracks.to_config()}
        if self.fixed is not None:
            config['fixed'] = self.fixed.to_config()
        config[self.regions.key] = self.regions.to_config()
        config['seed'] = self.seed
        return config

    @property
    def Y(self):
        return self.regions.Y

    @property
    def Y_M(self):
        return self.regions.Y_M

    def copy(self):
        import copy
        return copy.copy(self)

    def uid(self):
        return hex(hash(self.to_config()))[2:]

    def __len__(self):
        return len(self.regions)

    def __iter__(self):
        with features(self.tracks, self.fixed, self.regions.cell_lines()) as f:
            it = iter_regions(f, self.regions, n=self.n, randomize=self.randomize, infinite=self.infinite
                             , weights=self.weights, random_state=self.random_state, dtype=self.dtype)
            if self.chunk_size > 1:
                it = utils.chunks(it, size=self.chunk_size)
            for x in it:
                yield x

    def chunks(self, chunk_size):
        cpy = RegionsIterator(self.tracks, self.fixed, self.regions, randomize=self.randomize
                                , infinite=self.infinite, weights=self.weights, chunk_size=self.chunk_size, seed=self.seed, dtype=self.dtype)
        cpy.random_state = copy.deepcopy(self.random_state)
        return cpy

def iter_regions(feats, regions, n=None, randomize=False, infinite=False, weights=None, random_state=np.random.RandomState(), dtype=np.float32):
    if weights is not None and randomize and infinite:
        for x in load_regions(feats, weighted_sampler(regions, weights, random_state), n=n, dtype=dtype):
            yield x
    iterator = within_class_sampler(regions, random_state, infinite=infinite, randomize=randomize)
    for x in load_regions(feats, iterator, n=n, dtype=dtype):
        yield x

def interleave_balanced(x, y):
    n = len(x)
    m = len(y)
    threshold = n/m
    count = 0
    i = 0
    j = 0
    while i < n and j < m:
        if count >= threshold and j < m:
            yield y[j]
            j += 1
            count -= threshold
        elif i < n:
            yield x[i]
            i += 1
            count += 1
        else:
            yield y[j]
            j += 1

def within_class_sampler(regions, random_state, infinite=True, randomize=True):
    labels = np.any((regions.Y*regions.Y_M)==1, axis=1)
    pos = regions[labels]
    neg = regions[~labels]
    while infinite:
        if randomize:
            pos = pos[random_state.permutation(len(pos))] 
            neg = neg[random_state.permutation(len(neg))]
        for x in interleave_balanced(pos, neg):
            yield x
    if randomize:
        pos = pos[random_state.permutation(len(pos))] 
        neg = neg[random_state.permutation(len(neg))]
    for x in interleave_balanced(pos, neg):
        yield x

def weighted_sampler(regions, weights, random_state):
    labels = np.any((regions.Y*regions.Y_M)==1, axis=1)
    pos = regions[labels]
    neg = regions[~labels]
    regions = [neg, pos]
    while True:
        i = random_state.choice(2, p=weights)
        pop = regions[i]
        j = random_state.choice(len(pop))
        yield pop[j]

def load_regions(feats, regions, n=None, dtype=np.float32):
    if n is None:
        n, m = max_regions(regions)
    for R in regions:
        if type(R) is tuple:
            r, Y, Y_M = R
        else:
            Y = None
        X, F = feats[r]
        X = X.astype(dtype)
        F = F.astype(dtype)
        X_M = np.ones((n,1), dtype=dtype)
        if len(X) < n:
            X_ = np.zeros((n,X.shape[-1]), X.dtype)
            X_[:len(X)] = X
            X_M[len(X):] = 0
            X = X_
        extras = ()
        if Y is not None:
            Y = Y.reshape((len(Y), 1))
            Y_M = Y_M.reshape((len(Y_M), 1))
            extras = (Y.astype(dtype), Y_M.astype(dtype))
        yield (X, X_M, F) + extras

def max_region(regions):
    n = max(r[3]-r[2] for r in regions)
    return n
    







