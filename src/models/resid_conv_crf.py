import numpy as np

from src.models.resid_conv_logit import ResidConvLogitModel
from rnn.theano.crf import CRF

class ResidConvCRF(object):
    def __init__(self, units=64, layers=2, **kwargs):
        self.units = units
        self.layers = layers
        self.name = 'resid_conv_crf'
        self.kwargs = kwargs

    def to_config(self):
        config = {'layers': self.layers, 'units': self.units}
        config.update(self.kwargs)
        return [self.name, config]

    def __call__(self, shape, dtype='float32', random=np.random):
        n_in, _, n_fixed = shape
        return ResidConvCRFModel(self.units, self.layers, n_in, n_fixed, dtype, random, **self.kwargs)

class ResidConvCRFModel(ResidConvLogitModel):
    def __init__(self, *args, **kwargs):
        super(ResidConvCRFModel, self).__init__(*args, **kwargs)
        
    def make_predictor(self, units, dtype, random, use_bias):
        return CRF(units, 2, dtype=dtype, random=random, use_bias=use_bias)

    def predict(self, Z):
        ## Z is (batch, length, shape) but CRF expects (length, batch, shape)
        Z = Z.dimshuffle(1, 0, 2)
        logP = self.predict_layer.posterior(Z)
        return logP.dimshuffle(1, 0, 2)

    @property
    def shape(self):
        return 2 #output shape is 2 
