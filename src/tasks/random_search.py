
import sys
import os

def sample_params(dist, nsamples):
    import pymc3 as pm
    import theano
    theano.config.compute_test_value = 'ignore'
    
    old_stdout = sys.stdout
    print('# Sampling hyperparameters', file=sys.stderr)
    sys.stderr.flush()
    sys.stdout = sys.stderr
    model = pm.Model()
    thinning = 20
    with model:
        dist()
        start = pm.find_MAP()
        step = pm.NUTS(scaling=start)
        trace = pm.sample(thinning*nsamples, step, start=start)
    samples = trace[::thinning] 
    sys.stdout = old_stdout

    return samples

def random_search(f, dist, datas, nsamples):
    samples = sample_params(dist, nsamples)
    for sample in samples:
        for data in datas:
            yield f(sample, data)


