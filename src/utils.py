import numpy as np

def chunks(stream, size=1024):
    xs = []
    for x in stream:
        xs.append(x)
        if len(xs) == size:
            chunks = [np.stack(zs, axis=-2) for zs in zip(*xs)]
            yield size, chunks
            xs = []
    if len(xs) > 0:
        chunks = [np.stack(zs, axis=-2) for zs in zip(*xs)]
        yield len(xs), chunks
