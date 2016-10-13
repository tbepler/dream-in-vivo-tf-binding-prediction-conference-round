from __future__ import division

import math
import random
import numpy as np
import pandas as pd
import pickle
import sys

def null_func(*args):
    pass

class MinibatchModel(object):
    def __init__(self, minibatch_size=64, max_iters=500000, dtol=0.2, ttol=1e-6, chunk_size=None, **kwargs):
        if 'config' in kwargs:
            minibatch_size = config.get('minibatch_size', minibatch_size)
            max_iters = config.get('max_iters', max_iters)
            dtol = config.get('dtol', dtol)
            ttol = config.get('ttol', ttol)
            chunk_size = config.get('chunk_size', None)
            if chunk_size == 'default':
                chunk_size = None
        self.minibatch_size= minibatch_size
        self.max_iters = max_iters
        self.dtol = dtol
        self.ttol = 1e-6
        if chunk_size is None:
            self.chunk_size = minibatch_size
        else:
            self.chunk_size = chunk_size

    def to_config(self):
        config = {}
        config['minibatch_size'] = self.minibatch_size
        config['max_iters'] = self.max_iters
        config['dtol'] = self.dtol
        config['ttol'] = self.ttol
        config['chunk_size'] = self.chunk_size
        return config

    def snapshot(self, it, loss=None, path_prefix=None):
        cache = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        if path_prefix is not None:
            tokens = [path_prefix]
            it = 'iter{:06d}'.format(it)
            tokens.append(it)
            if loss is not None:
                loss = 'loss{:.5f}'.format(loss)
                tokens.append(loss)
            path = '_'.join(tokens) + '.pi'
            with open(path, 'wb') as f:
                f.write(cache)
        return cache

    def restore_state(self, state):
        unpickled = pickle.loads(state)
        self.__dict__.update(unpickled.__dict__)

    def prefetcher(self, stream, use=None):
        return stream

    def chunks(self, data):
        if hasattr(data, 'chunks'):
            return data.chunks(self.chunk_size)
        import src.utils as utils
        return utils.chunks(data, self.chunk_size)

    def fit(self, train, validate=None, callback=null_func, snapshot_prefix=None):
        return FitIterator(self, train, validate=validate, callback=callback, snapshot_prefix=snapshot_prefix)
        """
        if self.report_interval is None and validate is not None:
            report_interval = (len(validate) + self.chunk_size//2) // self.chunk_size
        elif self.report_interval is not None:
            report_interval = self.report_interval
        else:
            report_interval = 100
        train_loss = 0
        train_weight = 0
        train_summary = 0
        count = 0
        #history = [float('inf') for _ in range(self.early_stop_window)]
        history = 0
        best_loss = float('inf')
        best_model_state = None
        it = 0
        done = False
        chunks = self.chunks(train)
        for b, data in chunks:
            if it >= self.max_iters or done:
                break
            callback(it/self.max_iters, 'fit')
            it += 1
            #fit this chunk
            loss, weight, summary = self.fit_partial(*data)
            if np.isnan(loss):
                print('Error: NaN loss detected on training, iteration', it)
                done = True
            train_weight += weight
            train_loss += (loss-train_loss*weight)/train_weight
            train_summary += summary
            count += b
            if it % report_interval == 0:
                #compute the loss since last check
                if validate is not None:
                    loss, validation_summary = self.loss(validate, callback=callback)
                else:
                    loss = train_loss
                #check for early termination
                if loss <= best_loss*self.threshold:
                    history = 0
                else:
                    history += 1
                if history >= self.early_stop_window:
                    done = True
                #history.append(loss)
                #smoothed_loss_history = np.mean(history[-self.early_stop_window:])
                #if smoothed_loss_history - loss <= self.loss_tolerance:
                #    break
                if np.isnan(loss):
                    print('Error: NaN loss detected on validation, iteration', it)
                    done = True
                #check the loss against the best and snapshot current state if better
                if best_loss > loss:
                    best_model_state = self.snapshot(it, loss, path_prefix=snapshot_prefix)
                    best_loss = loss
                #generate training report and yield it
                train_summary = self.summarize(train_loss, train_summary)
                if validate is not None:
                    validation_summary.insert(0, 'Data', 'Valid')
                    train_summary.insert(0, 'Data', 'Train')
                    df = pd.concat([train_summary, validation_summary], axis=0)
                else:
                    df = train_summary
                df.insert(0, 'Iter', it*((self.chunk_size+self.minibatch_size-1)//self.minibatch_size))
                df.insert(1, 'Count', count)
                yield df
                train_summary = 0
                train_loss = 0
                train_weight = 0
        #restore state from the best model state
        if best_model_state is not None:
            self.restore_state(best_model_state)
        """

    def loss(self, data, callback=null_func):
        n = len(data)
        loss = 0
        weight = 0
        results = 0
        progress = 0.0
        callback(progress/n, 'loss')
        chunks = self.chunks(data)
        for b, x in chunks:
            part_loss, part_weight, part_results = self.loss_partial(*x)
            weight += part_weight
            loss += (part_loss - loss*part_weight)/weight
            results += part_results
            if np.isnan(loss):
                break
            progress += b
            callback(progress/n, 'loss')
        return loss, self.summarize(loss, results)

import array
import statistics
import math
import copy
import scipy.stats

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

class FitIterator(object):
    def __init__(self, model, train, validate=None, callback=null_func, snapshot_prefix=None):
        self.model = model
        self.train = train
        self.validate = validate
        self.callback = callback
        self.snapshot_prefix = snapshot_prefix
        self.it = 0
        self.history = array.array('d')
        self.train_stats = RollingStats(20)
        self.val_stats = RollingStats(20)
        self.next_report = 1
        self.report_interval = 0
        self.start()

    def start(self):
        self.train_stream = self.model.fit_stream(self.train)
        if self.validate is not None:
            self.loss_stream = self.model.loss_stream(self.validate)
        else:
            self.loss_stream = None

    def is_done(self):
        return self.model.optimizer.learning_rate < self.model.ttol or self.it >= self.model.max_iters

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_done():
            raise StopIteration()
        while self.it < self.next_report:
            self.it += 1
            ###RUN THE NEXT OPTIMIZATION STEP AND RECORD RESULTS
            loss, weight, result = next(self.train_stream)
            if np.isnan(loss):
                raise Exception('Error: NaN loss detected on training, iteration {}'.format(it))
            self.train_stats.next(loss/weight, result)
            if self.loss_stream is None:
                self.history.append(loss)
            else:
                loss, weight, result = next(self.loss_stream)
                if np.isnan(loss):
                    raise Exception('Error: NaN loss detected on validation, iteration {}'.format(it))
                self.history.append(loss)
                self.val_stats.next(loss/weight, result)
            ###CHECK FOR LEARNING RATE DECAY CONDITIONS
            n = len(self.history)//2
            if n >= 20:
                #use rank sum test of whether second half of chain is significantly less than first half
                z, p_val = scipy.stats.wilcoxon(self.history[:-n], self.history[-n:])
                #make one tail p-value for whether second half is less than first
                p_val /= 2
                if z < 0:
                    p_val = 1-p_val
                if p_val > self.model.dtol:
                    #shrink learning rate and report now
                    self.model.optimizer.learning_rate /= math.sqrt(10)
                    self.report_interval = 0
                    self.history = array.array('d')
                    break
                """
                #Geweke's convergence criteria between first half and last half of values
                mu0 = statistics.mean(self.history[:-n])
                var0 = statistics.variance(self.history[:-n])
                mu1 = statistics.mean(self.history[-n:])
                var1 = statistics.variance(self.history[-n:])
                z = (mu0-mu1)/math.sqrt(var0+var1)
                if z < self.model.dtol:
                    # shrink the learning rate and report now
                    self.model.optimizer.learning_rate /= math.sqrt(10)
                    self.report_interval = 0
                    self.history = array.array('d')
                    break
                """
            ###CHECK THE TERMINATION CRITERIA
            if self.is_done():
                break
        ###SNAPSHOT THE MODEL
        self.model.snapshot(self.it, path_prefix=self.snapshot_prefix)
        ###MAKE REPORT AND RETURN
        train_summary = self.model.summarize(*(self.train_stats.get()))
        if self.loss_stream is not None:
            validation_summary = self.model.summarize(*(self.val_stats.get()))
            validation_summary.insert(0, 'Data', 'Valid')
            train_summary.insert(0, 'Data', 'Train')
            df = pd.concat([train_summary, validation_summary], axis=0)
        else:
            df = train_summary
        df.insert(0, 'Iter', self.it)
        self.report_interval *= 2
        self.report_interval = max(self.report_interval, 1)
        self.report_interval = min(self.report_interval, 32)
        self.next_report = self.it + self.report_interval
        return df 



