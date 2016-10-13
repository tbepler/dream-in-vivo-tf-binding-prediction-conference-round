from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T

from rnn.theano.lstm import BLSTM
from rnn.theano.linear import Linear

class ReluLSTMLogit(object):
    def __init__(self, layers=[32, 32]):
        self.layers = layers
        self.name = 'relu_lstm_logit'

    def to_config(self):
        return ['relu_lstm_logit', {'layers': self.layers}]

    def __call__(self, n_in, n_fixed, dtype='float32', random=np.random):
        return ReluLSTMLogitModel(self.layers, n_in, n_fixed, dtype, random)

class ReluLSTM(object):
    def __init__(self, n_in, units, dtype, random):
        self.lstm = BLSTM(n_in, units*2, iact=T.nnet.sigmoid, fact=T.nnet.sigmoid
                         , oact=T.nnet.sigmoid, gact=T.tanh, cact=T.tanh, dtype=dtype
                         , random=random)
        self.linear = Linear(units*2, units, dtype=dtype, random=random)

    @property
    def weights(self):
        return self.lstm.weights + self.linear.weights

    def __call__(self, x, xm):
        z = self.lstm.scan(x, mask=xm)
        z = T.maximum(0, self.linear(z))
        return z

class ReluLSTMLogitModel(object):
    def __init__(self, layers, n_in, fixed_in, dtype, random):
        self.layers = layers
        self.dtype = dtype
        #model architecture
        self.relu_lstms = []
        for units in layers:
            f = ReluLSTM(n_in, units, dtype, random)
            self.relu_lstms.append(f)
            n_in = units
        self.fixed_linear = Linear(fixed_in, n_in, init_bias=1, dtype=self.dtype, random=random)
        self.W = theano.shared(np.ones(n_in, dtype=self.dtype))

    @property
    def weights(self):
        w = []
        for f in self.relu_lstms:
            w.extend(f.weights)
        w.extend(self.fixed_linear.weights)
        w.append(self.W)
        return w

    def __getstate__(self):
        state = self.__dict__.copy()
        state['W'] = self.W.get_value()
        return state

    def __setstate__(self, s):
        s['W'] = theano.shared(s['W'])
        self.__dict__.update(s)

    def __call__(self, X, X_M, F):
        A = X
        for f in self.relu_lstms:
            A = f(A, X_M)
        B = T.maximum(0, self.fixed_linear(F))
        Z = T.sum(A*B*self.W, axis=-1)
        Z_reshape = Z.dimshuffle(1, 'x', 0, 'x')
        filter_shape = (1, 1, 200, 1)
        Z = T.nnet.conv2d(Z_reshape, T.ones(filter_shape, dtype=X.dtype), filter_shape=filter_shape, subsample=(50, 1))
        Z = Z[:,0,:,0].dimshuffle(1, 0)
        return Z

    


        
