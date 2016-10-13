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
    def __init__(self, units, nu, gamma, rho, mom, l2):
        self.units = units
        self.nu = nu
        self.gamma = gamma
        self.rho = rho
        self.mom = mom
        self.l2 = l2

    def __call__(self):
        import pymc3 as pm
        names = ['Units']
        hypers = [pm.Poisson('Units', self.units)]
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
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, hypers):
        units = hypers['Units']
        nu = hypers['Nu']
        gamma = hypers['Gamma']
        rho = np.array(hypers['Rho'], dtype=theano.config.floatX)
        mom = np.array(hypers['Momentum'], dtype=theano.config.floatX)
        l2 = np.array(hypers['L2'], dtype=theano.config.floatX)
        decay = solvers.Annealing(gamma)
        opt = solvers.RMSprop(nu, decay=decay, rho=rho, momentum=mom)
        return LSTMReduceLogit(units, optimizer=opt, l2=l2, **self.kwargs)

class LSTMReduceLogit(BindingModel):
    def __init__(self, units, **kwargs):
        super(LSTMReduceLogit, self).__init__(200, 50, **kwargs)
        self.units = units

    def _setup(self, n_in, fixed_in):
        #model architecture
        layers = []
        units = self.units
        for i in range(4):
            lstml = LSTM(n_in, units, iact=T.nnet.sigmoid, fact=T.nnet.sigmoid
                         , oact=T.nnet.sigmoid, gact=T.tanh, cact=T.tanh)
            lstmr = LSTM(n_in, units, iact=T.nnet.sigmoid, fact=T.nnet.sigmoid
                         , oact=T.nnet.sigmoid, gact=T.tanh, cact=T.tanh)
            layers.append((lstml, lstmr))
            n_in = units
            units *= 2
        self.layers = layers
        self.fixed_linear = Linear(fixed_in, n_in, init_bias=1)
        self.logistic_regression = Linear(n_in, 1)

    @property
    def weights(self):
        w = []
        for (lstml, lstmr) in self.layers:
            w.extend(lstml.weights)
            w.extend(lstmr.weights)
        w.extend(self.fixed_linear.weights)
        w.extend(self.logistic_regression.weights)
        return w

    def __combine(self, Zl, Zr):
        return T.maximum(Zl, 0)*T.maximum(Zr, 0)

    def _predict_proba_graph(self, X, X_M, F, unroll=-1, **kwargs):
        #first layer reduces by 2
        lstml, lstmr = self.layers[0]
        Zl, _ = lstml.scanl(X, mask=X_M, unroll=unroll, **kwargs)
        Zr, _ = lstmr.scanr(X, mask=X_M, unroll=unroll, **kwargs)
        Zl = Zl[T.arange(1, Zl.shape[0], 2)]
        Zr = Zr[T.arange(0, Zr.shape[0], 2)]
        X_M = 1-(1-X_M.reshape((2, X_M.shape[0]//2, X_M.shape[1]))).prod(axis=0)
        #Z = T.concatenate((Zl, Zr), axis=-1)
        Z = self.__combine(Zl, Zr)
        unroll = unroll//2
        #second and third layers reduce by 5
        for i in [1,2]:
            lstml, lstmr = self.layers[i]
            Zl, _ = lstml.scanl(Z, mask=X_M, unroll=unroll, **kwargs)
            Zr, _ = lstmr.scanr(Z, mask=X_M, unroll=unroll, **kwargs)
            Zl = Zl[T.arange(4, Zl.shape[0], 5)]
            Zr = Zr[T.arange(0, Zr.shape[0], 5)]
            X_M = 1-(1-X_M.reshape((5, X_M.shape[0]//5, X_M.shape[1]))).prod(axis=0)
            #Z = T.concatenate((Zl, Zr), axis=-1)
            Z = self.__combine(Zl, Zr)
            unroll = unroll//5
        #fourth layer reduces by 4, stride is still 1 since regions are 200 every 50 bp
        lstml, lstmr = self.layers[3]
        Zl, _ = lstml.scanl(Z, mask=X_M, unroll=unroll, **kwargs)
        Zr, _ = lstmr.scanr(Z, mask=X_M, unroll=unroll, **kwargs)
        Zl = Zl[T.arange(3, Zl.shape[0])]
        Zr = Zr[T.arange(0, Zr.shape[0]-3)]
        Z = self.__combine(Zl, Zr)
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

    


        
