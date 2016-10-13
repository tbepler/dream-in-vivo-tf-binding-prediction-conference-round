
import sys
import pickle
import gzip
import importlib.machinery
import time
import json
import subprocess
import numpy as np
from contextlib import contextmanager

import src.device

class Config(object):
    @staticmethod
    @contextmanager
    def open(path):
        with open(path, 'w') as f:
            config = {}
            config['metadata'] = metadata()
            config['config'] = {}
            try:
                yield config['config']
            finally:
                dump(config, f)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def dump(obj, f):
    json.dump(obj, f, indent=4, sort_keys=True, cls=NumpyEncoder)

def datetime():
    return time.strftime('%Y-%m-%d-%H-%M-%S')

def gitversion():
    return subprocess.check_output(['git', 'describe', '--always'])[:-1].decode()

def metadata():
    md = {}
    md['git-version'] = gitversion()
    md['timestamp'] = datetime()
    main = sys.modules['__main__']
    if hasattr(main, '__file__'):
        main = main.__file__
    else:
        main = ''
    md['main'] = main
    md['args'] = sys.argv
    return md

def recursive_hash(o):
    if isinstance(o, dict):
        items = sorted(o.items())
        return recursive_hash(items)
    elif isinstance(o, (tuple, list)):
        hashes = tuple([recursive_hash(x) for x in o])
        return hash(hashes)
    return hash(o)

def uid(config):
    return recursive_hash(config)

def set_device(device):
    if device == 'gpu' or device == 'gpu_unused':
        #since theano is terrible at choosing an unused gpu
        #need to choose one here
        gpus = src.device.get_unused_gpus()
        while len(gpus) == 0:
            time.sleep(0.5) #spin until a gpu frees up
            gpus = src.device.get_unused_gpus()
        device = 'gpu{}'.format(gpus[0]) #just take the first free gpu
    if device.startswith('gpu'):
        import theano.sandbox.cuda
        theano.sandbox.cuda.use(device)

def load_model(config):
    source = config['source']
    params = config.get('params', {})
    if path.endswith('.pi.gz'):
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    elif path.endswith('.pi'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif path.endswith('.py'): #load from source code
        loader = importlib.machinery.SourceFileLoader('model_src', path)
        mod = loader.load_module()
        args = config.get('args', [])
        kwargs = config.get('kwargs', {})
        return mod.from_config(params)
    else: #load from class path
        tokens = source.split('.')
        mod = '.'.join(tokens[:-1])
        m = __import__(mod)
        for comp in tokens[1:]:
            m = getattr(m, comp)
        return m(config=params)

def from_model(model):
    params = model.to_config()
    source = model.__class__.__module__ + '.' + model.__class__.__name__
    return {'source': source, 'params': params}

def load_data(config):
    import src.loader
    return src.loader.from_config(config)

def from_data(data):
    return data.to_config()

def load(f):
    json.load(obj)
    



