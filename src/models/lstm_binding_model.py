from __future__ import print_function

import pandas as pd
import numpy as np
import theano
import theano.tensor as T

from src.models.minibatch_model import MinibatchModel
import rnn.theano.solvers as solvers
from rnn.theano.util import theano_compile
from rnn.theano.lstm import LayeredBLSTM, LSTM
from rnn.theano.linear import Linear

class LSTMBindingModel(MinibatchModel):
    def __init__(self, bin_width, stride, layers, optimizer=solvers.RMSprop(0.01), l2=0.01, **kwargs):
        super(LSTMBindingModel, self).__init__(**kwargs)
        self.bin_width = bin_width
        self.stride = stride
        self.layers = layers
        self.optimizer = optimizer
        self.l2 = l2
        self.is_setup = False
        
    def setup(self, n_in):
        if self.is_setup:
            return
        self.iters = 0
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
        self.logistic_regression = Linear(self.layers[-1], 1)
        self.is_setup = True

    @property
    def weights(self):
        w = []
        if self.blstm_stack is not None:
            w.extend(self.blstm_stack.weights)
        w.extend(self.left_encoder.weights)
        w.extend(self.right_encoder.weights)
        w.extend(self.logistic_regression.weights)
        return w

    @theano_compile(class_method=True)
    def __predict_proba(self, X, X_M):
        if self.blstm_stack is not None:
            X = self.blstm_stack.scan(X, mask=X_M)
        Z_l, _ = self.left_encoder.scanl(X, mask=X_M)
        Z_r, _ = self.right_encoder.scanr(X, mask=X_M)
        #bin the encoder values
        Z_l = Z_l[T.arange(self.bin_width-1, Z_l.shape[0], self.stride)]
        Z_r = Z_r[T.arange(0, Z_r.shape[0]-self.bin_width+1, self.stride)]
        #pool the binned encoder values
        Z = T.maximum(Z_l, Z_r)
        #logistic regression for bin probability
        lp = self.logistic_regression(Z)
        lp = T.flatten(lp, lp.ndim-1)
        #lp = T.reshape(lp, lp.shape[:-1])
        p = T.nnet.sigmoid(lp)
        return p

    @theano_compile(class_method=True)
    def __loss_partial(self, X, X_M, Y, Y_M):
        P = self.__predict_proba(X, X_M)
        cross_entropy = -Y*T.log(P) - (1-Y)*T.log(1-P)
        n = Y_M.sum()
        n = T.maximum(n, 1)
        L = T.sum(cross_entropy*Y_M)/n
        TP = T.sum((P>0.5)*Y*Y_M)
        FP = T.sum((P>0.5)*(1-Y)*Y_M)
        TN = T.sum((P<=0.5)*(1-Y)*Y_M)
        FN = T.sum((P<=0.5)*Y*Y_M)
        return L, TP, FP, TN, FN

    @theano_compile(updates=True, class_method=True)
    def __fit_partial(self, X, X_M, Y, Y_M, nu):
        L, TP, FP, TN, FN = self.__loss_partial(X, X_M, Y, Y_M)
        W = self.weights
        n = Y_M.sum()
        regularizer = self.l2*sum(T.sum(w**2) for w in W)*T.minimum(n, 1)/T.maximum(n, 1)
        G = T.grad(L+regularizer, self.weights) 
        updates = self.optimizer._updates(self.weights, G, nu)
        return [L, n, TP, FP, TN, FN], updates

    def fit_partial(self, X, Y):
        X, X_M = X
        Y, Y_M = Y
        self.setup(int(X.shape[-1]))
        nu = self.optimizer.decay(self.optimizer.learning_rate, iters=self.iters)
        nu = np.array(nu, dtype=theano.config.floatX)
        L, n, TP, FP, TN, FN = self.__fit_partial(X, X_M, Y, Y_M, nu)
        self.iters += 1
        return L, n, np.array([L*n, n, np.array([TP, FP, TN, FN])], dtype=object)

    def loss_partial(self, X, Y):
        X, X_M = X
        Y, Y_M = Y
        n = Y_M.sum()
        L, TP, FP, TN, FN = self.__loss_partial(X, X_M, Y, Y_M)
        return L, n, np.array([L*n, n, np.array([TP, FP, TN, FN])], dtype=object)

    def predict_proba_partial(self, X):
        X, X_M = X
        P = self.__predict_proba(X, X_M)
        return np.swapaxes(P, 0, 1)

    def summarize(self, arr):
        n = arr[1]
        L = arr[0]/n
        TP, FP, TN, FN = arr[2][0], arr[2][1], arr[2][2], arr[2][3]
        df = pd.DataFrame(arr[2].reshape(1,4), columns=['TP', 'FP', 'TN', 'FN'])
        df.insert(0, 'Loss', L)
        df.insert(1, 'Accuracy', (TP+TN)/n)
        df.insert(2, 'Sensitivity', TP/(TP+FN))
        df.insert(3, 'Specificity', TN/(FP+TN))
        df.insert(4, 'Precision', TP/(TP+FP))
        return df


    


        
