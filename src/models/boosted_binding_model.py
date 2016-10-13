import theano
import theano.tensor as T
import numpy as np

from src.models.binding_model import BindingModel, compile_mode, null_func

class BoostedBindingModel(BindingModel):
    def __init__(self, predictor, num_models=10, **kwargs):
        super(BoostedBindingModel, self).__init__(predictor, **kwargs)
        self.num_models = num_models

    def to_config(self):
        config = super(BoostedBindingModel, self).to_config()
        config['boosted'] = self.num_models
        return config

    def uid(self):
        name = self.predictor.name + '.boosted'
        uid = hex(hash(self.to_config()))[2:]
        return name + '.' + uid

    @property
    def shared(self):
        return self.models[-1].shared

    @property
    def weights(self):
        return self.models[-1].weights

    @property
    def bias(self):
        return self.models[-1].bias

    def setup(self, shape):
        #preprocessing normalization parameters
        self.preprocessor.setup(shape, dtype=self.dtype)

    def add_model(self, shape):
        model = self.predictor(shape, dtype=self.dtype, random=self.random)
        self.models.append(model)

    def _unscaled_predict(self, X):
        return sum(model(X) for model in self.models)

    def regularizer_penalty(self, n):
        W = self.weights # do not include bias terms
        one = np.cast[self.dtype](1)
        R = self.regularizer(W, dtype=self.dtype)*T.minimum(n, one)
        R += self.models[-1].regularizer()*T.minimum(n, one)
        return R

    def fit(self, X, Y, validate=None, callback=null_func, path_prefix=None, **kwargs):
        self.models = []
        self.setup(X.shape)
        if hasattr(X, 'fixed_array') and X.fixed_array is not None and hasattr(self.preprocessor, 'fit_fixed'):
            self.preprocessor.fit_fixed(X.fixed_array)
        data = self.make_sampler(X, Y, **kwargs)
        for i in range(self.num_models):
            self.add_model(self.preprocessor.shape)
            for df in self.optimizer(self, data, validate=validate, callback=callback, dtype=self.dtype, mode=compile_mode):
                it = df['Iter'].values[0]
                self.snapshot(it, path_prefix=path_prefix)
                df.insert(0, 'Model', i+1)
                yield df



