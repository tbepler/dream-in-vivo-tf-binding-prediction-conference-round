from __future__ import division

import pymc3 as pm
import theano
theano.config.compute_test_value = 'ignore'
import os
import sys
import pickle

def null_func(*args):
    pass

class GridSearch(object):
    def __init__(self, model_const, params, *args, **kwargs):
        self.model_const = model_const
        self.params = params
        self.args = args
        self.kwargs = kwargs

    def __getstate__(self):
        s = {}
        #fancy shmansy function serialization for model constructor
        import marshal
        code = marshal.dumps(self.model_const.func_code)
        closure = marshal.dumps(self.model_const.func_closure)
        s['model_const'] = (code, closure)

        s['params'] = self.params
        s['args'] = self.args
        s['kwargs'] = self.kwargs
        if self.__hasattr__('model'):
            s['model'] = self.model
        return s

    def __setstate__(self, s):
        #fancy shmansy function deserialization
        import marshal, types
        code, closure = s['model_const']
        code = marshal.loads(code)
        closure = marshal.loads(closure)
        self.model_const = types.FunctionType(code, globals(), closure=closure)

        self.params = s['params']
        self.args = s['args']
        self.kwargs = s['kwargs']
        if 'model' in s:
            self.model = s['model']

    def __getattr__(self, name):
        return self.model.__getattr__(name)

    def fit(self, train, validate=None, callback=null_func, snapshot_prefix=None):
        best_model = None
        best_loss = float('inf')
        if snapshot_prefix is not None:
            tokens = [snapshot_prefix]
            places = len(self.params)//10 + 1
            tokens.append('model{:0'+str(places)+'d}')
            for colname in list(self.params.columns):
                tokens.append(colname+'{}')
            prefix_template = '_'.join(tokens)
        for tup in self.params.itertuples():
            idx = tup[0]
            tup = tup[1:]
            model = self.model_const(tup, *self.args, **self.kwargs)
            if snapshot_prefix is not None:
                prefix = prefix_template.format(idx, *tup)
            else:
                prefix = None
            for df in model.fit(train, validate=validate, callback=callback
                                , snapshot_prefix=prefix):
                df.insert(0, 'Model', idx)
                i = 1
                for j, colname in enumerate(list(self.params.columns)):
                    df.insert(i, colname, tup[j])
                    i += 1
                yield df
            if validate is not None:
                loss, _ = model.loss(validate, callback=callback)
            else:
                #TODO fix this
                loss, _ = model.loss(X_train, Y_train, callback=callback)
            if loss < best_loss:
                self.best_model = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
                self.best_loss = loss
        self.model = pickle.loads(self.best_model)

def nCk(n,k): 
    return int( reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1) )

class RandomSearch(object):
    def __init__(self, model_const, prior, max_iters, tol=1e-5, *args, **kwargs):
        self.model_const = model_const
        self.prior = prior
        self.max_iters = max_iters
        self.tol = tol
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, name):
        return self.model.__getattr__(name)

    def sample_hypers(self):
        old_stdout = sys.stdout
        print('# Sampling hyperparameters', file=sys.stderr)
        sys.stderr.flush()
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            model = pm.Model()
            with model:
                names, hypers = self.prior()
                start = pm.find_MAP()
                step = pm.NUTS(scaling=start)
                trace = pm.sample(200, step, start=start, progressbar=False)
            trace = trace[-1]
        sys.stdout = old_stdout
        print('# Hypers:', trace, file=sys.stderr)
        sys.stderr.flush()
        return names, trace

    def stopping_criteria(self, losses):
        if len(losses) > 2:
            losses.sort()
            mu = losses[0]*(len(losses)-1) + losses[1]
            mu /= len(losses)
            n0 = nCk(len(losses),2)
            n1 = len(losses)-2
            mu2 = losses[0]*n0+losses[1]*n1+losses[2]
            mu2 /= (n0+n1+1)
            if mu+self.tol >= mu2:
                return True
        return False

    def fit(self, train, validate=None, callback=null_func, snapshot_prefix=None):
        best_model = None
        best_loss = float('inf')
        if snapshot_prefix is not None:
            tokens = [snapshot_prefix]
            places = self.max_iters//10 + 1
            tokens.append('M{:0'+str(places)+'d}')
            prefix_template = '_'.join(tokens)
        it = 0
        losses = []
        while True:
            if it >= self.max_iters:
                break
            names, hypers = self.sample_hypers()
            model = self.model_const(hypers, *self.args, **self.kwargs)
            if snapshot_prefix is not None:
                prefix = prefix_template.format(it)
            else:
                prefix = None
            for df in model.fit(train, validate=validate, callback=callback
                                , snapshot_prefix=prefix):
                df.insert(0, 'Model', it)
                for j, colname in enumerate(names):
                    df.insert(j+1, colname, hypers[colname])
                yield df
            if validate is not None:
                loss, _ = model.loss(validate, callback=callback)
            else:
                #TODO fix this
                loss, _ = model.loss(X_train, Y_train, callback=callback)
            if loss < best_loss:
                self.best_model = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
                self.best_loss = loss
            it += 1
            losses.append(loss)
            if self.stopping_criteria(losses):
                break
        self.model = pickle.loads(self.best_model)

