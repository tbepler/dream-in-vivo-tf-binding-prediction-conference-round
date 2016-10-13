import numpy as np

def shuffle(n, random=np.random):
    I = np.arange(n)
    while True:
        random.shuffle(I)
        for i in I:
            yield i

def stratified_shuffle(groups, weights=None, random=np.random):
    ## ensure groups contain indices not logicals
    index_groups = []
    for group in groups:
        if group.dtype == bool:
            group = np.flatnonzero(group)
        index_groups.append(group)
    groups = index_groups
    ## calculate the weight of each group
    if weights is None:
        weights = np.array([len(group) for group in groups])
    elif type(weights) == list:
        weights = np.array(weights)
    if np.all(weights == 0):
        weights = np.ones(len(weights), dtype=np.int8)
    assert len(weights) == len(groups)
    ## make the shuffled data streams
    index_streams = [shuffle(len(group), random=random) for group in groups]
    accum = np.zeros_like(weights)
    inc = weights.min()
    i = 0
    while True:
        ## draw index for group i
        j = next(index_streams[i])
        yield groups[i][j]
        ## increment accumulator for i
        ## and check for proceeding to next group
        accum[i] += inc
        if accum[i] >= weights[i]:
            accum[i] -= weights[i]
            i = (i+1)%len(groups)

def blocks(stream, block_size):
    assert block_size > 0
    x = next(stream)
    a = np.empty(block_size, x.dtype)
    a[0] = x
    i = 1%block_size
    if i == 0:
        yield a
    for x in stream:
        a[i] = x
        i = (i+1)%block_size
        if i == 0:
            yield a
    if i > 0:
        yield a[:i]

def sample_(data, num_samples=None, groups=None, weights=None, block_size=1, random=np.random):
    if type(data) is tuple:
        data = lazy_zip(data)
    if groups is not None:
        random_stream = stratified_shuffle(groups, weights=weights, random=random)
    else:
        random_stream = shuffle(len(data), random=random)
    if block_size > 1:
        random_stream = blocks(random_stream, block_size)
    if num_samples is None:
        for I in random_stream:
            yield data[I]
    else:
        count = 0
        while count < num_samples:
            I = next(random_stream)
            yield data[I]
            count += 1

def macrobatches(stream, size):
    import itertools
    while True:
        yield np.array(list(itertools.islice(stream, size)))

## BEGIN YUCK
mp_data = None ## this is for multiprocessing pool subprocesses to only get the data once... ugh
def mp_init_process(data):
    global mp_data
    mp_data = data

def mp_get_macrobatch(I):
    return mp_data[I]
## END YUCK

def stack(elements):
    elements = list(elements)
    stacks = [np.stack(x, axis=0) for x in zip(*elements)]
    return tuple(stacks)

def sample_stream(X, I):
    for i in I:
        yield i, X[i]

def macrobatch_sampler(X, Y=None, groups=None, weights=None, size=512, random=np.random, mp=False):
    if groups is not None:
        i_stream = stratified_shuffle(groups, weights=weights, random=random)
    else:
        i_stream = shuffle(len(X), random=random)
    index_batches = macrobatches(i_stream, size)
    if mp:
        stream = prefetch(sample_stream, X, i_stream, prefetch='mp', prefetch_buffer=2*size)
    else:
        stream = sample_stream(X, i_stream)
    import itertools
    while True:
        chunk = list(itertools.islice(stream, size))
        I_macro, X_macro = zip(*chunk)
        X_macro = stack(X_macro)
        if Y is not None:
            I_macro = np.array(list(I_macro))
            yield X_macro, Y[I_macro]
        else:
            yield X_macro

def sample(*args, **kwargs):
    if kwargs.get('prefetch', None) is not None:
        stream =  prefetch(sample_, *args, **kwargs)
    else:
        stream = sample_(*args, **kwargs)
    for x in stream:
        yield x

## loading data in a backround thread/process
def run_prefetch(Q, stop, f, *args, **kwargs):
    from queue import Full
    for x in f(*args, **kwargs):
        while not stop.is_set():
            try:
                Q.put(x, timeout=1)
                break
            except Full:
                pass
        if stop.is_set():
            break
    if not stop.is_set():
        Q.put(None)

def prefetch(*args, **kwargs):
    method = kwargs['prefetch']
    n = kwargs.get('prefetch_buffer', 1024)
    del kwargs['prefetch']
    if 'prefetch_buffer' in kwargs:
        del kwargs['prefetch_buffer']
    if method == 'thread':
        from threading import Thread, Event
        from queue import Queue
        Q = Queue(n)
        stop = Event()
        p = Thread(target=run_prefetch, args=(Q, stop)+args, kwargs=kwargs, daemon=True)
    elif method == 'mp':
        from multiprocessing import Process, Queue, Event
        Q = Queue(n)
        stop = Event()
        p = Process(target=run_prefetch, args=(Q, stop)+args, kwargs=kwargs, daemon=True)
    else:
        raise Exception("Unrecognized prefetch method: '{}' (valid options are 'thread', 'mp')".format(method))
    from queue import Empty
    p.start()
    try:
        while True:
            if not p.is_alive():
                raise Exception(p.exitcode)
            try:
                x = Q.get(timeout=1)
            except Empty:
                continue
            if x is None:
                break
            yield x
    finally:
        stop.set()
        ## flush the Q
        empty = False
        while not empty and p.is_alive():
            try:
                Q.get(timeout=0.25)
            except Empty:
                empty = True
        it = 0
        try:
            ## wait for thread to die
            while p.is_alive():
                p.join(timeout=1.0)
                it += 1
                if it > 5: #waited too long, thread is unresponsive so just go on, send terminate if not thread
                    if method == 'mp':
                        p.terminate()
                    break
        except AssertionError:
            #well this is awkward - finalizer was called from something other than main thread?
            pass

## lazy zip of indexable objects
class Zip(object):
    def __init__(self, objs):
        assert len(objs) > 0
        n = len(objs[0])
        assert all(len(obj)==n for obj in objs)
        self.objs = objs
        self.n = n

    def open(self):
        for obj in objs:
            if hasattr(obj, 'open'):
                obj.open()

    def close(self):
        for obj in objs:
            if hasattr(obj, 'close'):
                obj.close()

    def __len__(self):
        return self.n

    def __getitem__(self, I):
        return tuple([obj[I] for obj in self.objs])

def lazy_zip(args):
    return Zip(args)

## class for lazily indexing objects
class Select(object):
    def __init__(self, obj, I):
        self.obj = obj
        self.I = I

    def __len__(self):
        return len(self.I)

    def __getitem__(self, J):
        idxs = self.I[J]
        return self.obj[idxs]

def select(obj, I):
    return Select(obj, I)


