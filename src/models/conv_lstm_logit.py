from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T

from src.models.binding_model import BindingModel
import rnn.theano.solvers as solvers
from rnn.theano.lstm import LayeredBLSTM, LSTM
from rnn.theano.linear import Linear

def identity(x):
    return x

class Prior(object):
    def __init__(self, layers, nu, gamma, rho, mom, l2):
        self.layers = layers
        self.nu = nu
        self.gamma = gamma
        self.rho = rho
        self.mom = mom
        self.l2 = l2

    def __call__(self):
        import pymc3 as pm
        names = ['Layer{}'.format(i) for i in range(len(self.layers))]
        hypers = [pm.Poisson(name, n) for name,n in zip(names, self.layers)]
        nu = pm.Lognormal('Nu', np.log(self.nu)-0.5, tau=1)
        names.append('Nu')
        hypers.append(nu)
        gamma = pm.Poisson('Gamma', self.gamma)
        names.append('Gamma')
        hypers.append(gamma)
        rho = pm.Beta('Rho', alpha=2*self.rho/(1-self.rho), beta=2)
        names.append('Rho')
        hypers.append(rho)
        mom = pm.Beta('Momentum', alpha=2*self.mom/(1-self.mom), beta=2)
        names.append('Momentum')
        hypers.append(mom)
        l2 = pm.Lognormal('L2', np.log(self.l2)+1, tau=1)
        names.append('L2')
        hypers.append(l2)
        return names, hypers

class Constructor(object):
    def __init__(self, bin_width, stride, n_layers, **kwargs):
        self.bin_width = bin_width
        self.stride = stride
        self.n_layers = n_layers
        self.kwargs = kwargs

    def __call__(self, hypers):
        names = ['Layer{}'.format(i) for i in range(self.n_layers)]
        layers = [hypers[name] for name in names]
        nu = hypers['Nu']
        gamma = hypers['Gamma']
        rho = np.array(hypers['Rho'], dtype=theano.config.floatX)
        mom = np.array(hypers['Momentum'], dtype=theano.config.floatX)
        l2 = np.array(hypers['L2'], dtype=theano.config.floatX)
        decay = solvers.Annealing(gamma)
        opt = solvers.RMSprop(nu, decay=decay, rho=rho, momentum=mom)
        return LSTMLogit(self.bin_width, self.stride, layers, optimizer=opt, l2=l2, **self.kwargs)

class Conv(object):
    def __init__(self, n_in, n_out, layers):
        pass

class ConvLSTMLogit(BindingModel):
    def __init__(self, bin_width, stride, layers, **kwargs):
        super(LSTMLogit, self).__init__(**kwargs)
        self.layers = layers

    def __setup(self, n_in, fixed_in):
        #model architecture
        if len(self.layers) > 1:
            self.blstm_stack = LayeredBLSTM(n_in, self.layers[:-1], iact=T.nnet.sigmoid, fact=T.nnet.sigmoid
                                           , oact=T.nnet.sigmoid, gact=T.tanh, cact=T.tanh)
            n_in = self.layers[-2]
        else:
            self.blstm_stack = None
        self.left_encoder = LSTM(n_in, self.layers[-1], iact=T.nnet.sigmoid, fact=T.nnet.sigmoid
                                , oact=T.nnet.sigmoid, gact=T.tanh, cact=T.tanh)
        self.right_encoder = LSTM(n_in, self.layers[-1], iact=T.nnet.sigmoid, fact=T.nnet.sigmoid
                                 , oact=T.nnet.sigmoid, gact=T.tanh, cact=T.tanh)
        #self.left_encoder = LSTM(n_in, self.layers[-1], iact=T.nnet.sigmoid, fact=T.nnet.sigmoid
        #                        , oact=T.nnet.sigmoid, gact=identity, cact=identity)
        #self.right_encoder = LSTM(n_in, self.layers[-1], iact=T.nnet.sigmoid, fact=T.nnet.sigmoid
        #                         , oact=T.nnet.sigmoid, gact=identity, cact=identity)
        self.fixed_linear = Linear(fixed_in, self.layers[-1], init_bias=1)
        self.logistic_regression = Linear(self.layers[-1], 1)

    @property
    def weights(self):
        w = []
        if self.blstm_stack is not None:
            w.extend(self.blstm_stack.weights)
        w.extend(self.left_encoder.weights)
        w.extend(self.right_encoder.weights)
        w.extend(self.logistic_regression.weights)
        return w

    def __predict_proba_graph(self, X, X_M, F, unroll=-1):
        if self.blstm_stack is not None:
            X = self.blstm_stack.scan(X, mask=X_M, unroll=unroll)
        Z_l, _ = self.left_encoder.scanl(X, mask=X_M, unroll=unroll)
        Z_r, _ = self.right_encoder.scanr(X, mask=X_M, unroll=unroll)
        #bin the encoder values
        I = T.arange(self.bin_width-1, Z_l.shape[0], self.stride)
        Z_l = Z_l[I]
        #start = self.bin_width-1
        #end = Z_l.shape[0]
        #Z_l = Z_l[start:end][::self.stride]
        J = T.arange(0, Z_r.shape[0]-self.bin_width+1, self.stride)
        Z_r = Z_r[J]
        #start = 0
        #end = Z_r.shape[0]-self.bin_width+1
        #Z_r = Z_r[start:end][::self.stride]
        #pool the binned encoder values
        Z = T.maximum(Z_l, 0)*T.maximum(Z_r, 0)
        #compute the features from the fixed input
        V = T.maximum(self.fixed_linear(F), 0)
        #combine fixed and sequence features
        Z *= V
        #Z = T.exp(Z+V)
        #logistic regression for bin probability
        lp = self.logistic_regression(Z)
        lp = T.flatten(lp, lp.ndim-1)
        #lp = T.reshape(lp, lp.shape[:-1])
        p = T.nnet.sigmoid(lp)
        return p

    


        
