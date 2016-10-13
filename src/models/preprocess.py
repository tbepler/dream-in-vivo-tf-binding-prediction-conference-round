import theano
import theano.tensor as T
import numpy as np

def from_config(string):
    if string is 'zero_inflated_mean':
        return ZeroInflatedMean()
    if string is 'gaussian':
        return Gaussian()
    return Null()

def to_config(obj):
    if type(obj) == ZeroInflatedMean:
        return 'zero_inflated_mean'
    if type(obj) == Gaussian:
        return 'gaussian'
    return 'none'

class Null(object):
    def __init__(self):
        pass

    def setup(self, *args):
        pass

    def to_config(self):
        return 'none'

    def fit(self, *args):
        return (), []

    def preprocess(self, X, *args):
        return X


#this preprocessor
# standardize DNAse by z-score
# standardize RNA-seq by taking the log(x+c) where c is a small constant (to remove zeros)
#      then z-score normalizing
class DreamChallengePreprocessor(object):
    def __init__(self, c=1e-3):
        self.c = c
        self.is_setup = False

    def setup(self, shape, dtype=theano.config.floatX):
        n_in, _, fixed_in = shape
        self.x_mu = theano.shared(np.zeros(n_in-5, dtype=dtype))
        self.x_mu2 = theano.shared(np.zeros(n_in-5, dtype=dtype))
        self.x_obs = theano.shared(np.zeros(n_in-5, dtype=dtype))
        self.f_mu = 0
        self.f_sd = 0
        if fixed_in != 0:
            self.fixed_in = fixed_in
            fixed_in = fixed_in[-1]
            self.f_mu = theano.shared(np.zeros(fixed_in, dtype=dtype))
            self.f_sd = theano.shared(np.zeros(fixed_in, dtype=dtype))
        self.is_setup = True
        
    def to_config(self):
        return {'DNAse': 'z-score', 'RNA-seq': 'log-z-score'}

    def __getstate__(self):
        s = {}
        s['c'] = self.c
        if self.is_setup:
            s['x_mu'] = self.x_mu.get_value()
            s['x_mu2'] = self.x_mu2.get_value()
            s['x_obs'] = self.x_obs.get_value()
            if self.f_mu != 0:
                s['fixed_in'] = self.fixed_in
                s['f_mu'] = self.f_mu.get_value()
                s['f_sd'] = self.f_sd.get_value()
        return s

    def __setstate__(self, s):
        if 'c' in s:
            self.c = s['c']
        else:
            self.c = 1e-3
        if len(s) > 1:
            self.x_mu = theano.shared(s['x_mu'])
            self.x_mu2 = theano.shared(s['x_mu2'])
            self.x_obs = theano.shared(s['x_obs'])
            self.f_mu = 0
            self.f_sd = 0
            if 'f_mu' in s:
                self.fixed_in = s['fixed_in']
                self.f_mu = theano.shared(s['f_mu'])
                self.f_sd = theano.shared(s['f_sd'])
            self.is_setup=True
        else:
            self.is_setup=False

    @property
    def shape(self):
        track = 7
        fixed = 0
        if self.f_mu != 0:
            fixed = self.fixed_in
        return (track, 0, fixed)

    def __preprocess(self, X, X_M, F, x_mu, x_mu2, x_obs):
        x_obs = T.maximum(x_obs, 2)
        x_var = x_mu2/(x_obs-1)
        x_sd = T.sqrt(x_var)
        I = T.eq(x_sd, 0) | (x_obs < 2)
        ## this doesn't work with GPU optimizations in theano
        #x_sd = T.set_subtensor(x_sd[I.nonzero()], 1)
        x_sd += I

        ## log transform the RNA-seq values
        c = np.cast[X.dtype](self.c)
        X = T.set_subtensor(X[:,:,6:], T.log(X[:,:,6:]+c))

        ## subtract the mean and divide by the standard deviation
        X = T.set_subtensor(X[:,:,5:], (X[:,:,5:]-x_mu)/x_sd)

        if self.f_mu != 0:
            # using fixed vector
            # log-transform
            F = T.log(F+c)
            # z-score normalization
            F = (F-self.f_mu)/self.f_sd

        return (X, X_M, F)

    def fit_fixed(self, F):
        if self.f_mu != 0:
            F = np.log(F+self.c)
            f_mu = F.mean(0).astype(self.f_mu.dtype)
            self.f_mu.set_value(f_mu)
            f_sd = F.std(0).astype(self.f_sd.dtype)
            f_sd[f_sd==0] = 1
            self.f_sd.set_value(f_sd)

    def fit(self, X):
        X, X_M, F = X
        X_M = T.shape_padright(X_M)
        X = X[:,:,5:]
        ## log-transform the RNA-seq
        c = np.cast[X.dtype](self.c)
        X = T.set_subtensor(X[:,:,1:], T.log(X[:,:,1:]+c))
        ## update online mean and variance calculations
        x_obs = self.x_obs + X_M.sum(axis=[0,1]) #.astype(self.x_obs.dtype)
        delta = X - self.x_mu
        delta *= X_M
        x_mu = self.x_mu + delta.sum(axis=[0,1])/T.maximum(x_obs, 1)
        x_mu = x_mu.astype(self.x_mu.dtype)
        x_mu2 = self.x_mu2 + (delta*(X-x_mu)).sum(axis=[0,1])
        x_mu2 = x_mu2.astype(self.x_mu2.dtype)

        updates = [(self.x_mu, x_mu), (self.x_mu2, x_mu2), (self.x_obs, x_obs)]
        return (x_mu, x_mu2, x_obs), updates

    def preprocess(self, X, *args):
        X, X_M, F = X
        shared_args = [self.x_mu, self.x_mu2, self.x_obs]
        for i in range(len(args)):
            shared_args[i] = args[i]
        return self.__preprocess(X, X_M, F, *shared_args)

#this preprocessor
# standardize DNAse by dividing by the mean of the non-zero values
# standardize RNA-seq by z-score normalizing the log of non-zero values
class DreamChallengePreprocessor2(object):
    def __init__(self, use_rna_seq_track=True):
        self.is_setup = False
        self.use_rna_seq_track = use_rna_seq_track

    def setup(self, shape, dtype=theano.config.floatX):
        n_in, _, fixed_in = shape
        self.dnase_mu = theano.shared(np.zeros(1, dtype=dtype))
        self.dnase_obs = theano.shared(np.zeros(1, dtype=dtype))
        self.rna_mu = theano.shared(np.zeros(n_in-6, dtype=dtype))
        self.rna_mu2 = theano.shared(np.zeros(n_in-6, dtype=dtype))
        self.rna_obs = theano.shared(np.zeros(n_in-6, dtype=dtype))
        self.f_mu = 0
        self.f_sd = 0
        if fixed_in != 0:
            self.fixed_in = fixed_in
            fixed_in = fixed_in[-1]
            self.f_mu = theano.shared(np.zeros(fixed_in, dtype=dtype))
            self.f_sd = theano.shared(np.zeros(fixed_in, dtype=dtype))
        self.is_setup = True
        
    def to_config(self):
        return {'DNAse': 'mean', 'RNA-seq': 'exp-log-z-score'
                , 'use_rna_seq_track': self.use_rna_seq_track, 'Fixed': 'log-z-score'}

    def __getstate__(self):
        s = {}
        s['use_rna_seq_track'] = self.use_rna_seq_track
        if self.is_setup:
            s['dnase_mu'] = self.dnase_mu.get_value()
            s['dnase_obs'] = self.dnase_obs.get_value()
            s['rna_mu'] = self.rna_mu.get_value()
            s['rna_mu2'] = self.rna_mu2.get_value()
            s['rna_obs'] = self.rna_obs.get_value()
            if self.f_mu != 0:
                s['fixed_in'] = self.fixed_in
                s['f_mu'] = self.f_mu.get_value()
                s['f_sd'] = self.f_sd.get_value()
        return s

    def __setstate__(self, s):
        self.use_rna_seq_track = s.get('use_rna_seq_track', True)
        if 'dnase_mu' in s:
            self.dnase_mu = theano.shared(s['dnase_mu'])
            self.dnase_obs = theano.shared(s['dnase_obs'])
            self.rna_mu = theano.shared(s['rna_mu'])
            self.rna_mu2 = theano.shared(s['rna_mu2'])
            if isinstance(s['rna_obs'], np.ndarray):
                self.rna_obs = theano.shared(s['rna_obs'])
            else:
                self.rna_obs = s['rna_obs']
            self.f_mu = 0
            self.f_sd = 0
            if 'f_mu' in s:
                self.fixed_in = s['fixed_in']
                self.f_mu = theano.shared(s['f_mu'])
                self.f_sd = theano.shared(s['f_sd'])
            self.is_setup=True
        else:
            self.is_setup=False

    @property
    def shape(self):
        track = 6
        if self.use_rna_seq_track:
            track = 7
        fixed = 0
        if self.f_mu != 0:
            fixed = self.fixed_in
        return (track, 0, fixed)

    def standard_log_normal(self, x, mu, sd):
        # to make the re-exponentiated distribution be mean 1, variance 1 (for non-zero elements)
        # need to set the variance to ln(2) and the mean to -ln(2)/2 for the log'd values
        I = T.neq(sd, 0)
        delta = -mu - np.cast[x.dtype](np.log(2)/2)*I # set the mean to 0 if sd is 0
        mult = np.cast[x.dtype](np.sqrt(np.log(2)))**I/sd**I # set to 1 if sd is 0
        return (x+delta)*mult

    def __preprocess(self, X, X_M, F, dnase_mu, rna_mu, rna_mu2, rna_obs):
        ## DNAse normalization
        dnase_mu = T.maximum(dnase_mu, 1e-6)
        X = T.set_subtensor(X[:,:,5:6], X[:,:,5:6]/dnase_mu)

        ##RNA-seq normalization
        if self.use_rna_seq_track:
            rna_obs = T.maximum(rna_obs, 2)
            rna_var = rna_mu2/(rna_obs-1)
            rna_sd = T.sqrt(rna_var)
            log_rna = T.log(X[:,:,6:])
            log_rna_standardized = self.standard_log_normal(log_rna, rna_mu, rna_sd)
            X = T.set_subtensor(X[:,:,6:], T.exp(log_rna_standardized))
        else:
            X = X[:,:,:6]

        ## Fixed vector normalization
        if self.f_mu != 0:
            # using fixed vector
            # log-transform
            c = np.cast[F.dtype](1e-4)
            F = T.log(F+c)
            I = T.neq(self.f_sd, 0)
            F = (F-self.f_mu)/self.f_sd**I

        return (X, X_M, F)

    def fit_fixed(self, F):
        if self.f_mu != 0:
            c = np.cast[self.f_mu.dtype](1e-4)
            masked = np.log(F+c)
            f_mu = masked.mean(0).astype(self.f_mu.dtype)
            self.f_mu.set_value(f_mu)
            f_sd = masked.std(0).astype(self.f_sd.dtype)
            self.f_sd.set_value(f_sd)

    def fit(self, X):
        X, X_M, F = X
        X_M = T.shape_padright(X_M)

        ## fit the DNAse track
        DNAse = X[:,:,5:6]
        mask = T.neq(DNAse, 0)
        dnase_obs = self.dnase_obs + (X_M*mask).sum(axis=[0,1])
        delta = DNAse - self.dnase_mu
        delta *= X_M*mask
        dnase_mu = self.dnase_mu + delta.sum(axis=[0,1])/T.maximum(dnase_obs, 1)
        dnase_mu = dnase_mu.astype(self.dnase_mu.dtype)
        
        updates = [(self.dnase_mu, dnase_mu), (self.dnase_obs, dnase_obs)]
        results = (dnase_mu,)

        ## fit the RNA-seq tracks
        if self.use_rna_seq_track:
            RNAseq = X[:,:,6:]
            mask = T.neq(RNAseq, 0)
            rna_obs = self.rna_obs + (X_M*mask).sum(axis=[0,1])
            logRNAseq = T.log(RNAseq+1-mask) #add the (1-mask) to turn -inf into 0
            delta = logRNAseq - self.rna_mu
            delta *= X_M*mask
            rna_mu = self.rna_mu + delta.sum(axis=[0,1])/T.maximum(rna_obs, 1)
            rna_mu = rna_mu.astype(self.rna_mu.dtype)
            rna_mu2 = self.rna_mu2 + (delta*(logRNAseq-rna_mu)).sum(axis=[0,1])
            rna_mu2 = rna_mu2.astype(self.rna_mu2.dtype)
             
            updates += [(self.rna_mu, rna_mu), (self.rna_mu2, rna_mu2), (self.rna_obs, rna_obs)]
            results += (rna_mu, rna_mu2, rna_obs)

        return results, updates

    def preprocess(self, X, *args):
        X, X_M, F = X
        shared_args = [self.dnase_mu, self.rna_mu, self.rna_mu2, self.rna_obs]
        for i in range(len(args)):
            shared_args[i] = args[i]
        return self.__preprocess(X, X_M, F, *shared_args)

class ZeroInflatedMean(object):
    def __init__(self, use_rna_seq_track=True, c=1e-7):
        self.is_setup = False
        self.use_rna_seq_track = use_rna_seq_track
        self.c = c

    def setup(self, shape, dtype=theano.config.floatX):
        n_in, _, fixed_in = shape
        self.x_mu = theano.shared(np.zeros(n_in-5, dtype=dtype))
        self.x_obs = theano.shared(np.zeros(n_in-5, dtype=dtype))
        self.f_mu = 0
        self.f_obs = 0
        if fixed_in != 0:
            self.fixed_in = fixed_in
            fixed_in = fixed_in[-1]
            self.f_mu = theano.shared(np.zeros(fixed_in, dtype=dtype))
            self.f_obs = theano.shared(np.zeros(fixed_in, dtype=dtype))
        self.is_setup = True
        
    def to_config(self):
        return 'zero_inflated_mean'

    def __getstate__(self):
        s = self.__dict__.copy()
        if self.is_setup:
            s['x_mu'] = self.x_mu.get_value()
            s['x_obs'] = self.x_obs.get_value()
            if self.f_mu != 0:
                s['f_mu'] = self.f_mu.get_value()
                s['f_obs'] = self.f_obs.get_value()
        return s

    def __setstate__(self, s):
        self.__dict__.update(s)
        if 'x_mu' in s:
            self.x_mu = theano.shared(s['x_mu'])
            self.x_obs = theano.shared(s['x_obs'])
            if 'f_mu' in s:
                self.f_mu = theano.shared(s['f_mu'])
                self.f_obs = theano.shared(s['f_obs'])

    @property
    def shape(self):
        track = 6
        if self.use_rna_seq_track:
            track = 7
        fixed = 0
        if self.f_mu != 0:
            fixed = self.fixed_in
        return (track, 0, fixed)

    def __preprocess(self, X, X_M, F, x_mu, f_mu):
        x_mu = T.maximum(x_mu, np.cast[x_mu.dtype](1e-6))
        X_upd = T.set_subtensor(X[:,:,5:], X[:,:,5:]/x_mu)
        if not self.use_rna_seq_track:
            X_upd = X_upd[:,:,:6]
        F_upd = F
        if f_mu != 0:
            f_mu = T.maximum(f_mu, np.cast[f_mu.dtype](1e-6))
            F_upd = F/f_mu 
            F_upd = T.log(F_upd + np.cast[F.dtype](self.c))
        return (X_upd, X_M, F_upd)

    def fit_fixed(self, F):
        if self.f_mu != 0:
            mu = F.sum(0)/(F != 0).sum(0)
            mu[np.isnan(mu)] = 1
            mu = mu.astype(self.f_mu.dtype)
            self.f_mu.set_value(mu)

    def fit(self, X):
        X, X_M, F = X
        X_M = T.shape_padright(X_M)
        X = X[:,:,5:]
        nonzero = T.neq(X, 0).astype(X.dtype)
        x_obs = self.x_obs + (X_M*nonzero).sum(axis=[0,1]) #.astype(self.x_obs.dtype)
        delta = X - self.x_mu
        delta *= X_M*nonzero
        x_mu = self.x_mu + delta.sum(axis=[0,1])/T.maximum(x_obs, 1)
        x_mu = x_mu.astype(self.x_mu.dtype)
        updates = [(self.x_mu, x_mu), (self.x_obs, x_obs)]
        return (x_mu, self.f_mu), updates

    def preprocess(self, X, *args):
        X, X_M, F = X
        shared_args = [self.x_mu, self.f_mu]
        for i in range(len(args)):
            shared_args[i] = args[i]
        return self.__preprocess(X, X_M, F, *shared_args)


#TODO fix me
class Gaussian(object):
    def __init__(self):
        self.is_setup = False

    def setup(self, shape):
        n_in, _, fixed_in = shape
        self.x_mu = theano.shared(np.zeros(n_in-5, dtype=theano.config.floatX))
        self.x_mu2 = theano.shared(np.zeros(n_in-5, dtype=theano.config.floatX))
        self.x_obs = theano.shared(np.float32(0))
        self.f_mu = theano.shared(np.zeros(fixed_in, dtype=theano.config.floatX))
        self.f_mu2 = theano.shared(np.zeros(fixed_in, dtype=theano.config.floatX))
        self.f_obs = theano.shared(np.float32(0))
        self.is_setup = True

    def to_config(self):
        return 'gaussian'

    def __getstate__(self):
        s = {}
        if self.is_setup:
            s['x_mu'] = self.x_mu.get_value()
            s['x_mu2'] = self.x_mu2.get_value()
            s['x_obs'] = self.x_obs.get_value()
            s['f_mu'] = self.f_mu.get_value()
            s['f_mu2'] = self.f_mu2.get_value()
            s['f_obs'] = self.f_obs.get_value()
        return s

    def __setstate__(self, s):
        if len(s) > 0:
            self.x_mu = theano.shared(s['x_mu'])
            self.x_mu2 = theano.shared(s['x_mu2'])
            self.x_obs = theano.shared(s['x_obs'])
            self.f_mu = theano.shared(s['f_mu'])
            self.f_mu2 = theano.shared(s['f_mu2'])
            self.f_obs = theano.shared(s['f_obs'])
            self.is_setup=True
        else:
            self.is_setup=False

    def __preprocess(self, X, X_M, F, x_obs, x_mu, x_mu2, f_obs, f_mu, f_mu2):
        x_sd = T.sqrt(x_mu2/(x_obs-1))
        x_sd = T.maximum(x_sd, 1e-6)
        #x_sd = T.set_subtensor(x_sd[T.eq(x_sd,0).nonzero()], np.float32(1))
        X_upd = T.set_subtensor(X[:,:,5:], (X[:,:,5:] - x_mu)/x_sd)
        f_sd = T.sqrt(f_mu2/(f_obs-1))
        f_sd = T.maximum(f_sd, 1e-6)
        #f_sd = T.set_subtensor(f_sd[T.eq(f_sd,0).nonzero()], np.float32(1))
        F_upd = (F-f_mu)/f_sd
        return X_upd, F_upd

    def fit(self, X, X_M, F):
        X_M = T.shape_padright(X_M[...,0])
        X = X[:,:,5:]
        x_obs = self.x_obs + X_M.sum()
        delta = X - self.x_mu
        delta *= X_M
        x_mu = self.x_mu + delta.sum(axis=[0,1])/x_obs
        x_mu2 = self.x_mu2 + (delta*(X-x_mu)).sum(axis=[0,1])
        f_obs = self.f_obs + F.shape[0].astype(theano.config.floatX)
        delta = F - self.f_mu
        f_mu = self.f_mu + delta.sum(axis=0)/f_obs
        f_mu2 = self.f_mu2 + (delta*(F-f_mu)).sum(axis=0)
        updates = [(self.x_mu, x_mu), (self.x_mu2, x_mu2), (self.x_obs, x_obs), (self.f_mu, f_mu), (self.f_mu2, f_mu2), (self.f_obs, f_obs)]
        return (x_obs, x_mu, x_mu2, f_obs, f_mu, f_mu2), updates

    def preprocess(self, X, X_M, F, *args):
        shared_args = [self.x_obs, self.x_mu, self.x_mu2, self.f_obs, self.f_mu, self.f_mu2]
        for i in range(len(args)):
            shared_args[i] = args[i]
        return self.__preprocess(X, X_M, F, *shared_args)
