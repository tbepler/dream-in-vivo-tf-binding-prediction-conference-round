from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T

from rnn.theano.linear import Linear
from rnn.theano.lstm import LSTM
from rnn.initializers import orthogonal, xavier

from src.models.resid_conv_logit import BaseLayer, BottleneckResidConv, ResidConv, MaxPool, zeros

class ResidConvResidLSTM(object):
    def __init__(self, units=64, layers=2, pool=None, bottleneck=False, use_fixed=True
                , fixed_l1=0, fixed_l2=0.01):
        self.units = units
        self.layers = layers
        self.pool = pool
        self.bottleneck = bottleneck
        self.name = 'resid_conv_resid_lstm'
        self.use_fixed = use_fixed
        self.fixed_l1 = fixed_l1
        self.fixed_l2 = fixed_l2

    def to_config(self):
        config = {'layers': self.layers, 'units': self.units, 'pool': self.pool, 'bottleneck': self.bottleneck} 
        if hasattr(self, 'use_fixed'):
            config['use_fixed'] = self.use_fixed
            if hasattr(self, 'fixed_l1'):
                config['fixed_l1'] = self.fixed_l1
            if hasattr(self, 'fixed_l2'):
                config['fixed_l2'] = self.fixed_l2
        return [self.name, config]

    def __call__(self, shape, dtype='float32', random=np.random):
        n_in, _, n_fixed = shape
        kwargs = {'pool': self.pool, 'bottleneck': self.bottleneck}
        if hasattr(self, 'use_fixed'):
            kwargs['use_fixed'] = self.use_fixed
        if hasattr(self, 'fixed_l1'):
            kwargs['fixed_l1'] = self.fixed_l1
        if hasattr(self, 'fixed_l2'):
            kwargs['fixed_l2'] = self.fixed_l2
        return ResidConvResidLSTMModel(self.units, self.layers, n_in, n_fixed, dtype, random, **kwargs)

class CompressedResidLSTM(object):
    def __init__(self, n_in, units, dtype, random, stride=1, factor=4):
        self.stride = stride
        n_out = n_in // factor
        self.lstml = LSTM(n_in, n_out, iact=T.nnet.sigmoid, fact=T.nnet.sigmoid
                            , oact=T.nnet.sigmoid, gact=T.tanh, cact=T.tanh, dtype=dtype
                            , random=random)
        self.lstmr = LSTM(n_in, n_out, iact=T.nnet.sigmoid, fact=T.nnet.sigmoid
                            , oact=T.nnet.sigmoid, gact=T.tanh, cact=T.tanh, dtype=dtype
                            , random=random)
        self.linear = Linear(n_out*2, units)
        self.W = None
        if n_in != units:
            self.W = Linear(n_in, units, dtype=dtype, random=random)

    @property
    def shared(self):
        ws = self.lstml.shared + self.lstmr.shared + self.linear.shared
        if self.W is not None:
            ws += self.W.shared
        return ws

    @property
    def weights(self):
        ws = self.lstml.weights + self.lstmr.weights + self.linear.weights
        if self.W is not None:
            ws += self.W.weights
        return ws

    @property
    def bias(self):
        ws = self.lstml.bias + self.lstmr.bias + self.linear.bias
        if self.W is not None:
            ws += self.W.bias
        return ws

    def __call__(self, x, xm):
        ## note: lstm requires dimensions to be (length, batch, features)
        ## but x is input as (batch, feature, length, empty)
        x = x[...,0].dimshuffle(2, 0, 1)
        if self.stride != 1:
            x = x[::self.stride]
        ## it's unclear what to do with xm here
        zl, _ = self.lstml.scanl(x)
        zr, _ = self.lstmr.scanr(x)
        z = T.concatenate([zl, zr], axis=-1)
        if self.W is not None:
            x = self.W(x)
        z = self.linear(z)
        y = T.maximum(0, x+z)
        ## reshape y back to input shape
        return y.dimshuffle(1, 2, 0, 'x')

class ResidConvResidLSTMModel(object):
    def __init__(self, units, num_layers, n_in, fixed_in, dtype, random, pool='max', bottleneck=False
                , use_fixed=True, fixed_l2=0, fixed_l2=0.01):
        if type(num_layers) == int:
            num_layers = [num_layers]*4
        self.units = units
        self.num_layers = num_layers
        self.dtype = dtype
        if fixed_in == 0:
            use_fixed = False
        else:
            fixed_count, fixed_in = fixed_in
            self.fixed_l1 = np.cast[dtype](fixed_l1)
            self.fixed_l2 = np.cast[dtype](fixed_l2/fixed_count**2) ## apply a pretty heavy regularizer to the fixed weights if few fixed samples
        self.use_fixed = use_fixed
        if bottleneck:
            C = BottleneckResidConv
        else:
            C = ResidConv
        #model architecture
        self.base = BaseLayer(n_in, units, dtype, random)
        self.layers = []
        ##compute a stack of residual convolutions
        for _ in range(num_layers[0]):
            conv = C(units, units, dtype, random, width=3)
            self.layers.append(conv)
        ##thin by 2 with either convolution max pool or strided convolution
        if pool == 'max':
            conv = MaxPool(2)
            self.layers.append(conv)
            conv = C(units, 2*units, dtype, random, width=3)
        else:
            conv = C(units, 2*units, dtype, random, width=3, stride=2)
        self.layers.append(conv)
        units *= 2
        ##compute another stack of residual convolutions
        for _ in range(num_layers[1]):
            conv = C(units, units, dtype, random, width=3)
            self.layers.append(conv)
        ##thin by 5 with either LSTM max pool or strided LSTM 
        if pool == 'max':
            lstm = MaxPool(5)
            self.layers.append(lstm)
            lstm = CompressedResidLSTM(units, 2*units, dtype, random)
        else:
            lstm = CompressedResidLSTM(units, 2*units, dtype, random, stride=5)
        self.layers.append(lstm)
        units *= 2
        ##compute residual lstm stack 
        for _ in range(num_layers[2]):
            lstm = CompressedResidLSTM(units, units, dtype, random)
            self.layers.append(lstm)
        ##thin by 5 again
        if pool == 'max':
            lstm = MaxPool(5)
            self.layers.append(lstm)
            lstm = CompressedResidLSTM(units, 2*units, dtype, random)
        else:
            lstm = CompressedResidLSTM(units, 2*units, dtype, random, stride=5)
        self.layers.append(lstm)
        units *= 2
        ##compute another residual lstm stack 
        for _ in range(num_layers[3]):
            lstm = CompressedResidLSTM(units, units, dtype, random)
            self.layers.append(lstm)
        ##final fully connected layer per bin
        self.fc_shape = (units, units, 4, 1)
        fc = random.randn(*self.fc_shape).astype(dtype)
        xavier(fc)
        self.fc = theano.shared(fc)
        self.fc_bias = theano.shared(np.zeros(units, dtype=dtype))
        if self.use_fixed:
            self.fixed_linear = Linear(fixed_in, units, dtype=self.dtype, random=random, init=zeros)
        self.logit = Linear(units, 1, use_bias=False, dtype=self.dtype, random=random)

    @property
    def weights(self):
        w = self.base.weights
        for conv in self.convs:
            w.extend(conv.weights)
        w.append(self.fc)
        if hasattr(self, 'fixed_linear'):
            w.extend(self.fixed_linear.weights)
        w.extend(self.logit.weights)
        return w

    @property
    def bias(self):
        b = self.base.bias
        for conv in self.convs:
            b.extend(conv.bias)
        if hasattr(self, 'fc_bias'):
            b.append(self.fc_bias)
        if hasattr(self, 'fixed_linear'):
            b.extend(self.fixed_linear.bias)
        b.extend(self.logit.bias)
        return b

    @property
    def shared(self):
        s = self.base.shared
        for conv in self.convs:
            s.extend(conv.shared)
        s.append(self.fc)
        if hasattr(self, 'fc_bias'):
            s.append(self.fc_bias)
        if hasattr(self, 'fixed_linear'):
            s.extend(self.fixed_linear.shared)
        s.extend(self.logit.shared)
        return s 

    def __getstate__(self):
        state = self.__dict__.copy()
        state['fc'] = self.fc.get_value()
        state['fc_bias'] = self.fc_bias.get_value()
        return state

    def __setstate__(self, s):
        s['fc'] = theano.shared(s['fc'])
        if 'fc_bias' in s:
            s['fc_bias'] = theano.shared(s['fc_bias'])
        self.__dict__.update(s)
        if 'fixed_regularizer' in s:
            self.fixed_l2 = s['fixed_regularizer']

    def regularizer(self):
        R = 0
        if hasattr(self, 'fixed_l1'):
            R += self.fixed_l1*sum(T.sum(abs(w)) for w in self.fixed_linear.weights)
        if hasattr(self, 'fixed_l2'):
            R += self.fixed_l2*sum(T.sum(w**2) for w in self.fixed_linear.weights)/2
        return R

    def __call__(self, X):
        X, X_M, F = X ## X shape is (batch, length, features)
        X = X.dimshuffle(0, 2, 1, 'x')
        Z = self.base(X, X_M)
        ##layer stack transform
        for layer in self.layers:
            Z = layer(Z, X_M)
        ##final fully connected layer per bin
        Z = T.nnet.conv2d(Z, self.fc, filter_shape=self.fc_shape)
        #Z = T.maximum(0, Z)
        Z = Z[...,0].dimshuffle(0, 2, 1)
        if hasattr(self, 'fc_bias'):
            Z += self.fc_bias
        if hasattr(self, 'fixed_linear'):
            ## this way of combining the fixed effect can cause NaN
            ## due to the exponent
            # B = self.fixed_linear(F).dimshuffle(0, 'x', 1)
            # B = T.minimum(abs(B), T.exp(B))
            # Z *= B
            ## a more principled way
            ## F is already log transformed by the preprocessor
            # linear transformation of log(F) represents potential complexes
            B = self.fixed_linear(F).dimshuffle(0, 'x', 1)
            # add to Z
            Z += B
        Z = T.maximum(0, Z)
        ##linear transform for final value
        return self.logit(Z)[...,0]

    


        
