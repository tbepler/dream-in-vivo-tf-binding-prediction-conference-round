from __future__ import division, print_function

import pandas as pd
import numpy as np
import theano
import theano.tensor as T
import copy
import random
import pickle

import src.models.preprocess as preprocess
import src.models.optimizer as opt

from rnn.theano.util import theano_compile

compile_mode = theano.compile.mode.Mode(linker='cvm', optimizer='fast_run')
#compile_mode = theano.compile.mode.Mode(optimizer='fast_compile')

def null_func(*args, **kwargs):
    pass

class Regularizer(object):
    def __init__(self, l1=0, l2=0.001):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, W, dtype=theano.config.floatX):
        l1 = np.cast[dtype](self.l1)
        l2 = np.cast[dtype](self.l2)
        return l1*sum(T.sum(abs(w)) for w in W) + l2*sum(T.sum(w**2) for w in W)/2

    def to_config(self):
        return self.__dict__

class BindingModel(object):
    def __init__(self, predictor, optimizer=opt.RMSprop(), regularizer=Regularizer()
                , preprocessor=preprocess.ZeroInflatedMean(), class_weights=None
                , seed=None, dtype=theano.config.floatX):
        self.predictor = predictor
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.preprocessor = preprocessor
        self.class_weights = class_weights
        if seed is None:
            seed = random.getrandbits(32)
        self.seed = seed
        self.random = np.random.RandomState(seed)
        self.dtype = dtype
        self.loss = Loss(self)
        self.predict_proba = Predict(self)

    def to_config(self):
        config = {}
        config['predictor'] = self.predictor.to_config()
        config['optimizer'] = self.optimizer.to_config()
        config['regularizer'] = self.regularizer.to_config()
        config['preprocessor'] = self.preprocessor.to_config()
        if self.class_weights is not None:
            config['class_weights'] = self.class_weights
        config['seed'] = self.seed
        config['dtype'] = self.dtype
        return config

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def uid(self):
        name = self.predictor.name
        uid = hex(hash(self.to_config()))[2:]
        return name + '.' + uid

    @property
    def shared(self):
        return self.model.shared

    @property
    def weights(self):
        return self.model.weights

    @property
    def bias(self):
        return self.model.bias

    def setup(self, shape):
        #preprocessing normalization parameters
        self.preprocessor.setup(shape, dtype=self.dtype)
        self.model = self.predictor(self.preprocessor.shape, dtype=self.dtype, random=self.random)

    def snapshot(self, it, loss=None, path_prefix=None):
        if path_prefix is not None:
            tokens = [path_prefix]
            it = 'iter{:06d}'.format(it)
            tokens.append(it)
            if loss is not None:
                loss = '_loss{:.5f}'.format(loss)
                tokens.append(loss)
            path = ''.join(tokens) + '.pi'
            with open(path, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def summarize(self, L, arr):
        cross_entropy = arr[0]
        arr = arr[1:].astype('int64')
        TP, FP, TN, FN = arr[0], arr[1], arr[2], arr[3]
        n = arr.sum()
        df = pd.DataFrame(arr.reshape(1,4), columns=['TP', 'FP', 'TN', 'FN'])
        df.insert(0, 'Loss', L)
        df.insert(1, 'Accuracy', (TP+TN)/n)
        df.insert(2, 'Sensitivity', TP/(TP+FN))
        df.insert(3, 'Specificity', TN/(FP+TN))
        df.insert(4, 'Precision', TP/(TP+FP))
        df.insert(1, 'Cross Entropy', cross_entropy/n)
        return df

    def chunks(self, data, chunk_size=512):
        if hasattr(data, 'chunks'):
            return data.chunks(chunk_size)
        import src.utils as utils
        return utils.chunks(data, chunk_size)

    def _unscaled_predict(self, X):
        return self.model(X)

    @property
    def shape(self):
        return self.model.shape

    def _predict_proba(self, X, updates=False):
        args = []
        if updates:
            args, upd = self.preprocessor.fit(X)
        X = self.preprocessor.preprocess(X, *args)
        Z = self._unscaled_predict(X)
        ## check the model shape
        if self.shape > 1: ## outputs are per-class log probabilities
            P = Z
        else:
            P = T.nnet.sigmoid(Z)
        if updates:
            return P, upd
        return P

    def _Predict__predict_proba(self, *args):
        P = self._predict_proba(*args)
        if self.shape > 1:
            P = T.exp(P)[...,1]
        return P

    def _loss(self, X, Y, updates=False):
        mask = Y[...,1]
        Y = Y[...,0]
        if updates:
            P, upd = self._predict_proba(X, updates=updates)
        else:
            P = self._predict_proba(X, updates=updates)
        if self.shape > 1:
            ppos = P[...,1]
            npos = P[...,0]
            cross_entropy = -Y*ppos - (1-Y)*npos
            TP = (ppos>npos)*Y
            FP = (ppos>npos)*(1-Y)
            TN = (ppos<=npos)*(1-Y)
            FN = (ppos<=npos)*Y
        else:
            one = np.cast[self.dtype](1)
            cross_entropy = -Y*T.log(P) - (one-Y)*T.log(one-P)
            TP = (P>0.5)*Y
            FP = (P>0.5)*(one-Y)
            TN = (P<=0.5)*(one-Y)
            FN = (P<=0.5)*Y
        n = mask.sum()
        unweighted_cross_entropy = T.sum(cross_entropy*mask)
        if hasattr(self, 'loss_weight'):
            class_weight = T.constant(self.loss_weight)
            cross_entropy *= class_weight[0]**(1-Y) * class_weight[1]**Y
        L = T.sum(cross_entropy*mask)
        TP = T.sum(TP*mask)
        FP = T.sum(FP*mask)
        TN = T.sum(TN*mask)
        FN = T.sum(FN*mask)
        C = T.stack((unweighted_cross_entropy, TP, FP, TN, FN)).astype('float64')
        if updates:
            return (L, n, C), upd
        return L, n, C

    def _Loss__loss(self, *args):
        return self._loss(*args)

    def regularizer_penalty(self, n):
        W = self.weights ## don't penalize the bias terms
        one = np.cast[self.dtype](1)
        R = self.regularizer(W, dtype=self.dtype)*T.minimum(n, one) #multiply by 0 if no unmasked examples
        R += self.model.regularizer()*T.minimum(n, one)
        return R

    def _loss_grad(self, X, Y):
        (L, n, C), updates = self._loss(X, Y, updates=True)
        W = self.shared
        one = np.cast[self.dtype](1)
        R = self.regularizer_penalty(n)
        loss = L/T.maximum(one, n) + R
        G = T.grad(loss, W) 
        return (L, n, C, G, W), updates

    def __call__(self, data, updates=True, shapes=None):
        ## data is expected to be (X, Y) tuple
        if updates:
            (l, n, c, g, W), updates = self._loss_grad(*data)
            return {'loss': l, 'weight': n, 'extras': c, 'grad': g, 'W': W, 'updates': updates}
        else:
            L, n, C = self._loss(*data)
            return {'loss': L, 'weight': n, 'extras': C}
        
    def make_sampler(self, X, Y, sampler='stratified', macrobatch_size=None, seed=None, mp=False, config={}):
        import random
        ## use stratified sampling
        assert sampler == 'stratified'
        from src.models.sampler import macrobatch_sampler
        pos = Y[...,0]*Y[...,1]
        if pos.ndim > 1:
            pos = np.any(pos == 1, axis=1)
        groups = [pos, ~pos]
        if seed is None:
            seed = random.getrandbits(32)
        config['seed'] = seed
        config['sampler'] = sampler
        if macrobatch_size is None:
            macrobatch_size = self.optimizer.minibatch_size*8
        config['macrobatch_size'] = macrobatch_size
        data = macrobatch_sampler(X, Y=Y, groups=groups, random=np.random.RandomState(config['seed']), mp=mp)
        return data

    def fit(self, X, Y, validate=None, callback=null_func, path_prefix=None, **kwargs):
        self.setup(X.shape)
        if hasattr(X, 'fixed_array') and X.fixed_array is not None and hasattr(self.preprocessor, 'fit_fixed'):
            self.preprocessor.fit_fixed(X.fixed_array)
        if self.class_weights == 'balanced':
            n = Y[...,1].sum()
            num_pos = (Y[...,0]*Y[...,1]).sum()
            num_neg = n - num_pos
            self.loss_weight = np.empty(2, dtype=self.dtype)
            self.loss_weight[0] = n/num_neg
            self.loss_weight[1] = n/num_pos
            self.loss_weight *= 2/self.loss_weight.sum()
        elif self.class_weights is not None:
            self.loss_weight = self.class_weights
        data = self.make_sampler(X, Y, **kwargs)
        for df in self.optimizer(self, data, validate=validate, callback=callback, dtype=self.dtype, mode=compile_mode):
            it = df['Iter'].values[0]
            self.snapshot(it, path_prefix=path_prefix)
            yield df

class Loss(object):
    def __init__(self, model, b=64):
        self.model = model
        self.b = b

    def __getstate__(self):
        return {'model': self.model, 'b': self.b}

    def __setstate__(self, s):
        self.model = s['model']
        self.b = s['b']

    def __call__(self, *args):
        if not hasattr(self, 'f'):
            self.shared = make_shared(args, dtype=self.model.dtype) ## *args gives a tuple so this works
            i = T.iscalar()
            minis = slice_shared(self.shared, i, self.b)
            ys = self.model.__loss(*minis)
            self.f = theano.function([i], ys, mode=compile_mode)
        else:
            set_shared(self.shared, args, dtype=self.model.dtype)
        m = get_batch_size(args)
        L, n, C = 0, 0, 0
        for i in range(0, m, self.b):
            L_, n_, C_ = self.f(i)
            n += n_
            L += (L_ - L*n_)/n
            C += C_
        return model.summarize(L, C)

class Predict(object):
    def __init__(self, model, b=64):
        self.model = model
        self.b = b

    def __getstate__(self):
        return {'model': self.model, 'b': self.b}

    def __setstate__(self, s):
        self.model = s['model']
        self.b = s['b']

    def __call__(self, *args):
        if not hasattr(self, 'f'):
            self.shared = make_shared(args, dtype=self.model.dtype) ## *args gives a tuple so this works
            i = T.iscalar()
            minis = slice_shared(self.shared, i, self.b)
            ys = self.model.__predict_proba(*minis)
            self.f = theano.function([i], ys, mode=compile_mode)
        else:
            set_shared(self.shared, args, dtype=self.model.dtype)
        m = get_batch_size(args)
        ys = []
        for i in range(0, m, self.b):
            y = self.f(i)
            ys.append(y)
        return np.concatenate(ys, axis=0)

def get_batch_size(x):
    if type(x) == tuple:
        return get_batch_size(x[0])
    return x.shape[0]

def make_shared(x, dtype=None):
    if x is None:
        return 0
    if type(x) == tuple:
        return tuple([make_shared(y, dtype=dtype) for y in x])
    if dtype is None:
        dtype = x.dtype
    return theano.shared(x.astype(dtype), borrow=True)

def set_shared(shared, x, dtype=None):
    if shared == 0:
        return
    if type(shared) == tuple:
        for s,y in zip(shared, x):
            set_shared(s, y, dtype=dtype)
    else:
        if dtype is None:
            dtype = x.dtype
        shared.set_value(x.astype(dtype), borrow=True)

def slice_shared(shared, i, b):
    if shared == 0:
        return 0
    if type(shared) == tuple:
        return tuple([slice_shared(x, i, b) for x in shared])
    return shared[i:i+b]

