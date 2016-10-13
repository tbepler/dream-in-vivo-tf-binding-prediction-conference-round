from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.signal
import theano.tensor.signal.pool

from rnn.theano.linear import Linear
from rnn.initializers import orthogonal, xavier

def relu(x):
    return T.maximum(0, x)

def identity(x):
    return x

class ResidConvLogit(object):
    def __init__(self, units=64, layers=2, pool=None, bottleneck=False, use_fixed=True
                , fixed_l1=0, fixed_l2=0.01, use_resid_bias=False, use_bias=True, fixed_activation=identity):
        self.units = units
        self.layers = layers
        self.pool = pool
        self.bottleneck = bottleneck
        self.name = 'resid_conv_logit'
        self.use_fixed = use_fixed
        self.fixed_l1 = fixed_l1
        self.fixed_l2 = fixed_l2
        self.use_resid_bias = use_resid_bias
        self.use_bias = use_bias
        self.fixed_activation = fixed_activation

    def to_config(self):
        config = {'layers': self.layers, 'units': self.units, 'pool': self.pool, 'bottleneck': self.bottleneck}
        config['use_resid_bias'] = self.use_resid_bias
        config['use_bias'] = self.use_bias
        if hasattr(self, 'use_fixed'):
            config['use_fixed'] = self.use_fixed
            if hasattr(self, 'fixed_l1'):
                config['fixed_l1'] = self.fixed_l1
            if hasattr(self, 'fixed_l2'):
                config['fixed_l2'] = self.fixed_l2
            config['fixed_activation'] = self.fixed_activation.__name__
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
        if hasattr(self, 'use_resid_bias'):
            kwargs['use_resid_bias'] = self.use_resid_bias
        else:
            kwargs['use_resid_bias'] = True
        kwargs['use_bias'] = True
        if hasattr(self, 'use_bias'):
            kwargs['use_bias'] = self.use_bias
        kwargs['fixed_activation'] = identity
        if hasattr(self, 'fixed_activation'):
            kwargs['fixed_activation'] = self.fixed_activation
        return ResidConvLogitModel(self.units, self.layers, n_in, n_fixed, dtype, random, **kwargs)

class Conv(object):
    def __init__(self, n_in, units, shape, stride=None, border_mode='half', dtype='float32', random=np.random
                , use_bias=True):
        self.shape = shape
        self.stride = stride
        self.border_mode = border_mode
        w = random.randn(*shape).astype(dtype)
        xavier(w)
        self.w = theano.shared(w)
        if use_bias:
            b = np.zeros(units, dtype=dtype)
            self.b = theano.shared(b)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['w'] = state['w'].get_value()
        if 'b' in state:
            state['b'] = state['b'].get_value()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.w = theano.shared(self.w)
        if hasattr(self, 'b'):
            self.b = theano.shared(self.b)

    @property
    def shared(self):
        if hasattr(self, 'b'):
            return [self.w, self.b]
        return [self.w]

    @property
    def weights(self):
        return [self.w]

    @property
    def bias(self):
        if hasattr(self, 'b'):
            return [self.b]
        return []

    def __call__(self, x):
        kwargs = {'filter_shape': self.shape, 'border_mode': self.border_mode}
        if self.stride is not None:
            kwargs['subsample'] = self.stride
        y = T.nnet.conv2d(x, self.w, **kwargs)
        if hasattr(self, 'b'):
            y += self.b.dimshuffle(0, 'x', 'x')
        return y

class BaseLayer(object):
    def __init__(self, n_in, units, dtype, random, width=9, use_bias=True):
        conv_shape = (units, n_in, width, 1)
        self.conv = Conv(n_in, units, conv_shape, dtype=dtype, random=random, use_bias=use_bias)

    @property
    def shared(self):
        return self.conv.shared

    @property
    def weights(self):
        return self.conv.weights

    @property
    def bias(self):
        return self.conv.bias

    def __call__(self, x, xm):
        z = self.conv(x)
        return T.maximum(0, z)

class ResidConv(object):
    def __init__(self, n_in, units, dtype, random, width=3, stride=1, use_bias=True):
        conv1_shape = (units, n_in, width, 1)
        self.stride = (stride, 1)
        self.conv1 = Conv(n_in, units, conv1_shape, stride=self.stride, dtype=dtype, random=random
                         , use_bias=use_bias)
        conv2_shape = (units, units, width, 1)
        self.conv2 = Conv(units, units, conv2_shape, dtype=dtype, random=random, use_bias=use_bias)
        self.W = None
        if n_in != units:
            self.W = Linear(n_in, units, dtype=dtype, random=random)

    @property
    def shared(self):
        s = self.conv1.shared + self.conv2.shared
        if self.W is not None:
            s.extend(self.W.shared)
        return s

    @property
    def weights(self):
        ws = self.conv1.weights + self.conv2.weights
        if self.W is not None:
            ws.extend(self.W.weights)
        return ws

    @property
    def bias(self):
        b = self.conv1.bias + self.conv2.bias
        if self.W is not None:
            b.extend(self.W.bias)
        return b

    def __call__(self, x, xm):
        z = self.conv1(x)
        z = T.maximum(0, z)
        z = self.conv2(z)
        if self.stride != (1,1):
            x = x[:, :, T.arange(0, x.shape[2], self.stride[0])]
        if self.W is not None:
            x = x.dimshuffle(0, 2, 3, 1)
            x = self.W(x)
            x = x.dimshuffle(0, 3, 1, 2)
        return T.maximum(0, x+z)

class BottleneckResidConv(object):
    def __init__(self, n_in, units, dtype, random, width=3, stride=1, factor=4, use_bias=True):
        n = units//factor
        self.l1 = Linear(n_in, n, dtype=dtype, random=random)
        conv_shape = (n, n, width, 1)
        self.conv = Conv(n, n, conv_shape, dtype=dtype, random=random, use_bias=use_bias)
        self.l2 = Linear(n, units, dtype=dtype, random=random)
        ## projection if output units != input units
        self.W = None
        if n_in != units:
            self.W = Linear(n_in, units, dtype=dtype, random=random)
        self.stride = (stride, 1)

    @property
    def shared(self):
        ws = self.conv.shared 
        ws.extend(self.l1.shared)
        ws.extend(self.l2.shared)
        if self.W is not None:
            ws.append(self.W.shared)
        return ws

    @property
    def weights(self):
        ws = self.conv.weights 
        ws.extend(self.l1.weights)
        ws.extend(self.l2.weights)
        if self.W is not None:
            ws.append(self.W.weights)
        return ws

    @property
    def bias(self):
        ws = self.conv.bias
        ws.extend(self.l1.bias)
        ws.extend(self.l2.bias)
        if self.W is not None:
            ws.append(self.W.bias)
        return ws

    def __call__(self, x, xm):
        ## x is (batch, features, len, empty)
        if self.stride != (1,1):
            x = x[:, :, T.arange(0, x.shape[2], self.stride[0])]
        x = x.dimshuffle(0, 2, 3, 1)
        z = T.maximum(0, self.l1(x))
        z = z.dimshuffle(0, 3, 1, 2)
        z = self.conv(z)
        z = T.maximum(0, z)
        z = z.dimshuffle(0, 2, 3, 1)
        z = self.l2(z)
        z = z.dimshuffle(0, 3, 1, 2)
        if self.W is not None:
            x = self.W(x)
            x = x.dimshuffle(0, 3, 1, 2)
        return T.maximum(0, x+z)

class MaxPool(object):
    def __init__(self, width):
        self.width = width

    @property
    def weights(self):
        return []

    @property
    def bias(self):
        return []
    
    @property
    def shared(self):
        return []

    def __call__(self, x, xm):
        z = T.signal.pool.pool_2d(x, (self.width,1), ignore_border=False, st=(self.width,1), mode='max')
        return z

def zeros(w):
    w[:] = 0

class ResidConvLogitModel(object):
    def __init__(self, units, layers, n_in, fixed_in, dtype, random, pool=None, bottleneck=False, use_fixed=True
                , fixed_l1=0, fixed_l2=0.01, use_resid_bias=False, use_bias=True, fixed_activation=identity):
        if type(layers) == int:
            layers = [layers]*4
        self.units = units
        self.layers = layers
        self.dtype = dtype
        if fixed_in == 0:
            use_fixed = False
        else:
            fixed_count, fixed_in = fixed_in
            self.fixed_l1 = np.cast[dtype](fixed_l1)
            self.fixed_l2 = np.cast[dtype](fixed_l2/fixed_count**2) ## apply a pretty heavy regularizer to the fixed weights if few fixed samples
        self.use_bias = use_bias
        self.use_fixed = use_fixed
        self.use_resid_bias = use_resid_bias
        use_bias = use_resid_bias and use_bias
        if bottleneck:
            C = BottleneckResidConv
        else:
            C = ResidConv
        #model architecture
        self.base = BaseLayer(n_in, units, dtype, random, use_bias=self.use_bias)
        self.convs = []
        ## first convolutional stack
        for _ in range(layers[0]):
            conv = C(units, units, dtype, random, width=3, use_bias=use_bias)
            self.convs.append(conv)
        ## thin by 2
        if pool == 'max':
            conv = MaxPool(2)
            self.convs.append(conv)
            conv = C(units, 2*units, dtype, random, width=3, use_bias=use_bias)
        else:
            conv = C(units, 2*units, dtype, random, width=3, stride=2, use_bias=use_bias)
        self.convs.append(conv)
        units *= 2
        ## next stack
        for _ in range(layers[1]):
            conv = C(units, units, dtype, random, width=3, use_bias=use_bias)
            self.convs.append(conv)
        ## thin by 5
        if pool == 'max':
            conv = MaxPool(5)
            self.convs.append(conv)
            conv = C(units, 2*units, dtype, random, width=3, use_bias=use_bias)
        else:
            conv = C(units, 2*units, dtype, random, width=3, stride=5, use_bias=use_bias)
        self.convs.append(conv)
        units *= 2
        ## MOAR STACKS
        for _ in range(layers[2]):
            conv = C(units, units, dtype, random, width=3, use_bias=use_bias)
            self.convs.append(conv)
        ## thin by 5 again - now 4 values per bin
        if pool == 'max':
            conv = MaxPool(5)
            self.convs.append(conv)
            conv = C(units, 2*units, dtype, random, width=3, use_bias=use_bias)
        else:
            conv = C(units, 2*units, dtype, random, width=3, stride=5, use_bias=use_bias)
        self.convs.append(conv)
        units *= 2
        ## LAST STACK - STACKS... DO THEY DO THINGS? LETS FIND OUT
        for _ in range(layers[3]):
            conv = C(units, units, dtype, random, width=3, use_bias=use_bias)
            self.convs.append(conv)
        ## fully connected layer per bin - increase number of units 2-fold again
        self.fc_shape = (2*units, units, 4, 1)
        fc = random.randn(*self.fc_shape).astype(dtype)
        xavier(fc)
        self.fc = theano.shared(fc)
        if self.use_bias:
            self.fc_bias = theano.shared(np.zeros(2*units, dtype=dtype))
        if self.use_fixed:
            self.fixed_linear = Linear(fixed_in, 2*units, dtype=self.dtype, random=random, init=zeros
                                      , use_bias=False)
            self.fixed_activation = fixed_activation
        self.fc_activation = relu # T.tanh
        self.predict_layer = self.make_predictor(2*units, self.dtype, random, self.use_bias)

    def make_predictor(self, units, dtype, random, use_bias):
        return Linear(units, 1, dtype=dtype, random=random, use_bias=use_bias)

    def predict(self, Z):
        return self.predict_layer(Z)[...,0]

    @property
    def shape(self):
        return 1 #output shape is 1 

    @property
    def weights(self):
        w = self.base.weights
        for conv in self.convs:
            w.extend(conv.weights)
        w.append(self.fc)
        if hasattr(self, 'hidden'):
            w.extend(self.hidden.weights)
        if hasattr(self, 'fixed_linear'):
            w.extend(self.fixed_linear.weights)
        w.extend(self.predict_layer.weights)
        return w

    @property
    def bias(self):
        b = self.base.bias
        for conv in self.convs:
            b.extend(conv.bias)
        if hasattr(self, 'hidden'):
            b.extend(self.hidden.bias)
        if hasattr(self, 'fc_bias'):
            b.append(self.fc_bias)
        if hasattr(self, 'fixed_linear'):
            b.extend(self.fixed_linear.bias)
        b.extend(self.predict_layer.bias)
        return b

    @property
    def shared(self):
        s = self.base.shared
        for conv in self.convs:
            s.extend(conv.shared)
        s.append(self.fc)
        if hasattr(self, 'fc_bias'):
            s.append(self.fc_bias)
        if hasattr(self, 'hidden'):
            s.extend(self.hidden.shared)
        if hasattr(self, 'fixed_linear'):
            s.extend(self.fixed_linear.shared)
        s.extend(self.predict_layer.shared)
        return s 

    def __getstate__(self):
        state = self.__dict__.copy()
        state['fc'] = self.fc.get_value()
        if self.use_bias:
            state['fc_bias'] = self.fc_bias.get_value()
        return state

    def __setstate__(self, s):
        s['fc'] = theano.shared(s['fc'])
        if 'fc_bias' in s:
            s['fc_bias'] = theano.shared(s['fc_bias'])
        self.__dict__.update(s)
        if 'fixed_regularizer' in s:
            self.fixed_l2 = s['fixed_regularizer']
        if hasattr(self, 'logit'):
            if self.logit.ws.get_value().shape[0] == self.fc.get_value().shape[1]: #bias is NOT used
                self.logit.use_bias = False

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
        for conv in self.convs:
            Z = conv(Z, X_M)
        Z = T.nnet.conv2d(Z, self.fc, filter_shape=self.fc_shape)
        #Z = T.maximum(0, Z)
        Z = Z[...,0].dimshuffle(0, 2, 1)
        fc_activation = relu ## relu by default for backwards compatibility
        if hasattr(self, 'fc_activation'):
            fc_activation = self.fc_activation
        if hasattr(self, 'fc_bias'):
            Z += self.fc_bias
        if hasattr(self, 'hidden'): ## this is for backwards compatibility
            Z = T.maximum(0, Z)
            Zh = self.hidden(Z)
            if hasattr(self, 'fixed_linear'):
                Zh += self.fixed_linear(F).dimshuffle(0, 'x', 1)
            Z = T.tanh(Zh)
        elif hasattr(self, 'fixed_linear'):
            ## this way of combining the fixed effect can cause NaN
            ## due to the exponent
            # B = self.fixed_linear(F).dimshuffle(0, 'x', 1)
            # B = T.minimum(abs(B), T.exp(B))
            # Z *= B
            ## a more principled way
            ## F is already log transformed by the preprocessor
            # linear transformation of log(F) represents potential complexes
            B = self.fixed_linear(F).dimshuffle(0, 'x', 1)
            fixed_activation = identity #identity by default
            if hasattr(self, 'fixed_activation'):
                fixed_activation = self.fixed_activation
            B = fixed_activation(B)
            # add to Z
            Z += B
            Z = fc_activation(Z)
        else:
            Z = fc_activation(Z)
        return self.predict(Z)

    


        
