
import sys
import os
import multiprocessing as mp
import pandas as pd

def random_search(f, dist, partitions, nsamples, path_prefix, **kwargs):
    import src.tasks.random_search as rs

    def make_task(params, data):
        pass

    for task in rs.random_search(make_task, dist, partitions):
        pass

    decimal_places = nsamples//10 + 1
    prefix_template = path_prefix + '{:0' + str(decimal_places) + 'd}'
    def combinations():
        for i in range(nsamples):
            sample = samples[i]
            print(sample, file=sys.stderr)
            #sample = pm.trace_to_dataframe(samples[i])
            prefix = prefix_template.format(i)
            for (ident, train, validate) in partitions:
                print(ident, file=sys.stderr)
                for tag in ident.values[0]:
                    prefix += '.' + tag
                prefix += '/'
                if not os.path.exists(prefix):
                    os.makedirs(prefix)
                ident = ident.copy()
                ident.insert(0, 'Index', i)
                yield sample, train, validate, prefix, ident

    if njobs > 1:
        pool = mp.Pool(processes=njobs)
        generator = pool.imap_unordered(f, combinations())
    else:
        generator = (f(x) for x in combinations())
    first = True
    for df in generator:
        df.to_csv(sys.stdout, sep='\t', float_format='%.5f', header=first, index=False)
        first = False


