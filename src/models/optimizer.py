
import theano
import theano.tensor as T
import pandas as pd
import numpy as np
import array
import scipy.stats
import copy

import rnn.theano.sgd as sgd 

def null_func(*args):
    pass

def from_config(config):
    method = config['method']
    del config['method']
    if method == 'RMSprop':
        return RMSprop(**config)
    elif method == 'SGD':
        return SGD(**config)
    else:
        raise Exception('Unrecognized optimization method: {}'.format(method))

class Momentum(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def to_config(self):
        return {'base': self.kwargs}

    def __call__(self, dtype=theano.config.floatX):
        return sgd.Momentum(dtype=dtype, **self.kwargs)

class Nesterov(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def to_config(self):
        return {'nesterov': self.kwargs}

    def __call__(self, dtype=theano.config.floatX):
        return sgd.Nesterov(dtype=dtype, **self.kwargs)

class RMSprop(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def to_config(self):
        return {'rmsprop': self.kwargs}

    def __call__(self, dtype=theano.config.floatX):
        return sgd.RMSprop(dtype=dtype, **self.kwargs)

class SGD(object):
    def __init__(self, nu=0.01, decay=0, shrink_factor=np.sqrt(10), ntol=1e-6, momentum=None, conditioner=None, minibatch_size=64, convergence_sample=200, rtol=1e-5, max_iters=500000):
        self.nu = nu
        self.decay = decay
        self.momentum = momentum
        self.conditioner = conditioner
        self.minibatch_size = minibatch_size
        self.shrink_factor = shrink_factor
        self.ntol = ntol
        self.convergence_sample = convergence_sample
        self.rtol = rtol
        self.max_iters = max_iters

    def to_config(self):
        config = copy.deepcopy(self.__dict__)
        config['momentum'] = 'none' if self.momentum is None else self.momentum.to_config()
        config['conditioner'] = 'none' if self.conditioner is None else self.conditioner.to_config()
        return {'sgd': config}

    def solver(self, dtype=theano.config.floatX):
        M = None
        if self.momentum is not None:
            M = self.momentum(dtype=dtype)
        C = None
        if self.conditioner is not None:
            C = self.conditioner(dtype=dtype)
        return sgd.SGD(nu=self.nu, decay=self.decay, C=C, M=M, dtype=dtype)

    def convergence(self):
        coroutine = ranksums_convergence(self.p_tol)
        next(coroutine)
        return coroutine

    def make_report(self, model, it, train_stats, validate_stats=None, validate=False):
        train_summary = model.summarize(*train_stats)
        if validate_stats is not None:
            validation_summary = model.summarize(*validate_stats)
            validation_summary.insert(0, 'Data', 'Valid')
            train_summary.insert(0, 'Data', 'Train')
            df = pd.concat([train_summary, validation_summary], axis=0)
        elif validate:
            train_summary.insert(0, 'Data', 'Train')
            df = train_summary
        else:
            df = train_summary
        df.insert(0, 'Iter', it)
        return df

    def validation_blocks_stream(self, data, dtype=theano.config.floatX):
        if hasattr(data, '__len__'):
            ## validation data is finite
            ## in this case, simply iterate all the validation data each time
            ## validation needs to be performed
            while True:
                yield data
        else:
            ## validation data is infinite
            ## iterate only the convergence_sample number of minibatches
            ## each time validation needs to be performed
            import itertools
            num_elements = self.convergence_sample*self.minibatch_size
            yield blocks(itertools.islice(data, num_elements), self.chunk_size, dtype=dtype)

    def __call__(self, model, train, validate=None, callback=null_func, dtype=theano.config.floatX
                , theano_compile_opts={}, **kwargs):
        it = 0
        train_stats = AccumStats()
        ## TRAIN IS EXPECTED TO BE A MACROBATCH ITERATOR!!!!!
        #train = blocks(infinite(train), self.chunk_size, dtype=dtype)
        solver = self.solver(dtype=dtype)
        updates = update_steps(model, train, solver, self.minibatch_size, dtype, **theano_compile_opts)
        if validate is not None:
            ## VALIDATE MUST BE A MACROBATCH ITERATOR!!!!!
            losses = loss_stream(model, validate, self.minibatch_size, dtype, self.convergence_sample, **theano_compile_opts)
        #converge = self.convergence()
        next_report = 1
        report_interval = 0
        prev_loss = float('inf')
        done = False
        val_stats = None
        while it < self.max_iters and not done:
            it += 1
            loss, weight, result = next(updates) 
            if np.isnan(loss):
                raise Exception('Error: NaN loss detected on training, iteration {}'.format(it))
            train_stats.next(loss, result, count=weight)
            ## check for computing validation
            if validate is not None and it % self.convergence_sample == 0:
                val_stats = next(losses)
                if np.isnan(loss):
                    raise Exception('Error: NaN loss detected on validation, iteration {}'.format(it))
            #check for report
            if it >= next_report or it % self.convergence_sample == 0:
                df = self.make_report(model, it, train_stats.get(), validate_stats=val_stats, validate=(validate is not None))
                report_interval = max(report_interval*2, 1)
                report_interval = min(report_interval, self.convergence_sample)
                next_report = it + report_interval
                yield df
            #check convergence
            if it % self.convergence_sample == 0:
                cur_loss, _ = train_stats.pop()
                if val_stats is not None:
                    cur_loss, _ = val_stats
                r = (prev_loss-cur_loss)/prev_loss
                if not np.isnan(r) and r <= self.rtol:
                    if self.shrink_factor >= 1:
                        shrunk_nu = np.cast[dtype](solver.nu.get_value()/self.shrink_factor)
                        solver.nu.set_value(shrunk_nu)
                    else:
                        done = True
                prev_loss = cur_loss
                if self.shrink_factor >= 1:
                    nu = solver.lr.eval()
                    if nu <= self.ntol:
                        done = True
            """
            if converge.send(loss):
                shrunk_nu = np.cast[dtype](solver.nu.get_value()/self.shrink_factor)
                solver.nu.set_value(shrunk_nu)
                converge = self.convergence()
                next_report = it
                report_interval = 1
            nu = solver.lr.eval()
            """
        #df = self.make_report(model, it, train_stats.get(), validate_stats=(val_stats.get() if validate is not None else None))
        #yield df

def infinite(data):
    while True:
        for x in data:
            yield x

def make_blocks(x, block_size, dtype=None):
    if x is None:
        return None
    ##recursively make blocks to mirror tuple structure of x
    if type(x) == tuple:
        blocks = [make_blocks(y, block_size, dtype=dtype) for y in x]
        return tuple(blocks)
    if dtype is None:
        dtype = x.dtype
    block = np.zeros((block_size,)+x.shape, dtype=dtype)
    return block

def write_blocks(blocks, i, x):
    if blocks is None:
        return
    assert type(blocks) == type(x)
    if type(x) == tuple:
        for b,y in zip(blocks, x):
            write_blocks(b, i, y)
    else:
        blocks[i] = x

def slice_blocks(blocks, i):
    if blocks is None:
        return None
    if type(blocks) == tuple:
        return tuple([slice_blocks(b, i) for b in blocks])
    return blocks[:i]

def blocks(data, block_size, dtype=None):
    x = next(data)
    blocks = make_blocks(x, block_size, dtype=dtype)
    i = 0
    write_blocks(blocks, i, x)
    i = (i+1)%block_size
    if i == 0:
        yield block_size, blocks
    for x in data:
        write_blocks(blocks, i, x)
        i = (i+1)%block_size
        if i == 0:
            yield block_size, blocks
    if i != 0:
        yield i, slice_blocks(blocks, i)

def ranksums_convergence(p_tol, mi=16, ma=200):
    history = array.array('d')
    converged = False
    while not converged:
        loss = yield converged
        history.append(loss)
        ###CHECK FOR CONVERGENCE CONDITIONS
        n = len(history)//2
        if len(history) >= mi:
            n = min(len(history), ma)
            m = n//2
            x = history[-n:-m]
            y = history[-m:]
            #use rank sum test of whether second half of chain is significantly less than first half
            z, p_val = scipy.stats.ranksums(x, y)
            #make one tail p-value for whether second half is less than first
            p_val /= 2
            if z < 0:
                p_val = 1-p_val
            if p_val > p_tol:
                converged = True
    yield converged

def make_shared(x, dtype):
    if x is None:
        return 0
    if type(x) == tuple:
        return tuple([make_shared(y, dtype) for y in x])
    if dtype is not None:
        x = x.astype(dtype)
    return theano.shared(x, borrow=True)

def slice_shared(shared, i, b):
    if type(shared) == tuple:
        return tuple([slice_shared(x, i, b) for x in shared])
    if shared == 0:
        return shared
    return shared[i:i+b]

def write_shared(shared, x, dtype):
    if shared == 0:
        return
    if type(shared) == tuple:
        for s,y in zip(shared, x):
            write_shared(s, y, dtype)
    else:
        if dtype is not None:
            x = x.astype(dtype)
        shared.set_value(x, borrow=True)

def shapes(x):
    if x is None:
        return 0
    if type(x) == tuple:
        return tuple([shapes(y) for y in x])
    return x.shape[-1]

def num_examples(x):
    if x is None:
        return 0 #err... wot!
    if type(x) == tuple:
        return max(num_examples(z) for z in x)
    return x.shape[0]

## data is expected to be a macrobatch iterator
def update_steps(model, data, solver, minibatch_size, dtype, **kwargs):
    try:
        args = next(data)
        n = num_examples(args)
    except StopIteration:
        return
    ### COMPILE THE THEANO FUNCTION
    shared = make_shared(args, dtype)
    i = T.iscalar()
    b = minibatch_size
    minis = slice_shared(shared, i, b)
    res = model(minis, updates=True, shapes=shapes(args))
    W = res['W']
    if 'grad' in res:
        G = res['grad']
    else:
        G = T.grad(res['loss'], W)
    updates = res.get('updates', [])
    updates += solver.updates(W, G)
    res = (res['loss'], res.get('weight', 1), res.get('extras', 0))
    f = theano.function([i], res, updates=updates, **kwargs)
    ### fit the data stream
    for i in range(0, n, b):
        #nu = np.cast[theano.config.floatX](nu)
        results = f(i)
        yield results
    for args in data:
        n = num_examples(args)
        write_shared(shared, args, dtype)
        for i in range(0, n, b):
            results = f(i)
            yield results

def loss_stream(model, data, minibatch_size, dtype, minibatches_per_step, **kwargs):
    ## validate MAY BE a macrobatch iterator with finite length
    ## it could ALSO be infinite
    f = None
    b = minibatch_size
    stats = AccumStats()
    if hasattr(data, '__len__'):
        ## data is finite
        while True:
            for macrobatch in data:
                n = num_examples(macrobatch)
                if f is None:
                    shared = make_shared(macrobatch, dtype)
                    i = T.iscalar()
                    minis = slice_shared(shared, i, b)
                    res = model(minis, updates=False)
                    res = (res['loss'], res.get('weight', 1), res.get('extras', 0))
                    f = theano.function([i], res, **kwargs)
                else:
                    write_shared(shared, macrobatch, dtype)
                for i in range(0, n, b):
                    loss, weight, results = f(i)
                    stats.next(loss, results, count=weight)
            yield stats.pop()
    else:
        ## data is inifinite - need to use minibatches_per_step
        count = 0
        for macrobatch in data:
            n = num_examples(macrobatch)
            if f is None:
                shared = make_shared(macrobatch, dtype)
                i = T.iscalar()
                minis = slice_shared(shared, i, b)
                res = model(minis, updates=False)
                res = (res['loss'], res.get('weight', 1), res.get('extras', 0))
                f = theano.function([i], res, **kwargs)
            else:
                write_shared(shared, macrobatch, dtype)
            for i in range(0, n, b):
                loss, weight, results = f(i)
                stats.next(loss, results, count=weight)
                count += 1
                if count == minibatches_per_step:
                    yield stats.pop()
                    count = 0

class AccumStats(object):
    def __init__(self):
        self.loss = 0
        self.count = 0
        self.results = 0

    def next(self, loss, result, count=1):
        self.count += count
        self.loss += (loss - count*self.loss)/self.count
        self.results += result

    def get(self):
        return self.loss, self.results

    def pop(self):
        y = self.get()
        self.loss = 0
        self.count = 0
        self.results = 0
        return y

class RollingStats(object):
    def __init__(self, window):
        self.n = window
        self.i = 0
        self.count = 0
        self.loss = None
        self.results = None

    def next(self, loss, results):
        if self.loss is None:
            self.loss = np.zeros(self.n)
        if self.results is None:
            self.results = np.zeros((self.n, len(results)), dtype=results.dtype)
        self.loss[self.i] = loss
        self.results[self.i] = results
        self.count += 1
        self.i = (self.i+1)%self.n

    def get(self):
        if self.count < self.n:
            return np.median(self.loss[:self.count]), np.sum(self.results[:self.count], axis=0)
        return np.median(self.loss), np.sum(self.results, axis=0)


