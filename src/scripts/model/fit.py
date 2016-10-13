#!/usr/bin/env python

"""
This script takes a model file and training data and fits the model 
"""

from __future__ import print_function

import sys
import os
import pickle
import random
import numpy as np

from src.progress import Progress

def make_sampler(data, config={}, macrobatch_size=512):
    if data is None:
        return data
    X, Y = data
    ## use stratified sampling
    from src.models.sampler import sample, macrobatch_sampler
    pos = np.any((Y[...,0]*Y[...,1]) == 1, axis=1)
    groups = [pos, ~pos]
    config['seed'] = random.getrandbits(32)
    config['sampler'] = 'stratified'
    config['macrobatch_size'] = macrobatch_size
    data = macrobatch_sampler(X, Y=Y, groups=groups, random=np.random.RandomState(config['seed']), mp=True)
    return data

def fit(path_prefix, model, training, macrobatch_size=None, validate=None, report_out=sys.stdout, progress_out=sys.stderr, progress=None, config={}):
    if macrobatch_size is None:
        macrobatch_size = model.optimizer.minibatch_size*8
    #make data sampler for validation data if given
    config['validate'] = {}
    validate = make_sampler(validate, macrobatch_size=macrobatch_size, config=config['validate'])
    #make the model snapshots dir
    basename_prefix = os.path.basename(path_prefix)
    dirpath = path_prefix+'snapshots'
    config['snapshots'] = dirpath
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    snapshot_prefix = os.path.join(dirpath, basename_prefix)
    #fit the model
    if progress is None:
        progress = Progress(out=progress_out)
    X_train, Y_train = training
    config['train'] = {}
    reports = model.fit(X_train, Y_train, validate=validate, callback=progress.progress_monitor(), path_prefix=snapshot_prefix, mp=True, config=config['train'])
    first = True
    for df in reports:
        df.to_csv(report_out, sep='\t', float_format='%.5f', header=first, index=False) 
        report_out.flush()
        first = False
    #save the final model
    model_path = path_prefix+'model.pi'
    config['model'] = model_path
    with open(model_path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)








