import pickle
import gzip
import importlib.machinery

def load(path):
    if path.endswith('.py'):
        loader = importlib.machinery.SourceFileLoader('model_src', path)
        mod = loader.load_module()
        return mod.model()
    elif path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)
    



