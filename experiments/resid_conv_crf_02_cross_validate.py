
import numpy as np
import pandas as pd
import copy
import os
import sys
import random

curdir = os.path.dirname(__file__)
rootdir = os.path.dirname(curdir)
sys.path.insert(0, rootdir)
sys.path.insert(1, os.path.join(rootdir, 'rnn'))

class Partition(object):
    def __init__(self, tracks, fixed, labels, heldout, nbins=0, stride=1):
        self.tracks = tracks
        self.fixed = fixed
        self.labels = labels
        self.heldout = heldout
        self.nbins = nbins
        self.stride = stride
        TF = os.path.basename(labels).split('.')[0]
        self.ident = pd.Series()
        self.ident['Train'] = TF
        self.ident['Heldout'] = self.heldout

    def __call__(self):
        import src.dataset as dataset
        ## load the data
        config = {}
        config['labels'] = {}
        config['labels']['path'] = self.labels
        regions, Y = dataset.load_regions(self.labels)
        if self.nbins > 0: 
            config['labels']['slice'] = {'nbins': self.nbins, 'stride': self.stride}
            regions, Y = dataset.slice_regions(regions, self.nbins, self.stride, Y=Y) ## this can easily cause explosive memory usage if stride is too low...
        X = dataset.load_features(self.tracks, self.fixed, regions)
        I = X.cell_line == self.heldout 
        X_train = X.select(~I)
        ## this wasn't really a good idea...
        #if X_train.fixed is not None:
        #    X_train.fixed.df = X_train.fixed.df.drop(self.heldout, 1) #remove from fixed to get a correct test of preprocessor as well
        Y_train = Y[~I]
        X_validate = X.select(I)
        Y_validate = Y[I]
        config['train'] = {}
        config['train']['ident'] = self.ident['Train']
        config['train']['tracks'] = X_train.tracks.to_config()
        if X_train.fixed is not None:
            config['train']['fixed'] = X_train.fixed.to_config()
        config['heldout'] = {}
        config['heldout']['ident'] = self.heldout 
        config['heldout']['tracks'] = X_validate.tracks.to_config()
        if X_validate.fixed is not None:
            config['heldout']['fixed'] = X_validate.fixed.to_config()
        return config, (X_train, Y_train), (X_validate, Y_validate)

def make_partitions(tracks, fixed, labels, nbins, stride, config={}):
    import src.dataset as dataset
    config['labels'] = {}
    config['labels']['path'] = labels
    if nbins > 0:
        config['labels']['slice'] = {'nbins': nbins, 'stride': stride}
    config['tracks'] = tracks 
    config['fixed'] = fixed
    config['partition'] = 'leave_one_cell_line_out'
    for path in labels:
        TF = os.path.basename(path).split('.')[0]
        regions, y = dataset.load_regions(path)
        cell_lines = sorted(list(set(regions[:,0])))
        for cell_line in cell_lines:
            yield Partition(tracks, fixed, path, cell_line, nbins=nbins, stride=stride)
        
class ModelConstructor(object):
    def __init__(self, device='cpu', dtype='float32'):
        self.device = device
        self.dtype = dtype

    def __call__(self):
        import src.config as cfg
        cfg.set_device(self.device)
        return model_constructor(dtype=self.dtype)

    def to_config(self, config={}):
        model_constructor_config(config=config, dtype=self.dtype)

def model_constructor(dtype='float32'):
    from src.models.resid_conv_crf import ResidConvCRF
    predictor = ResidConvCRF(units=64, layers=[3, 6, 6, 3], pool='max', use_bias=False
                              , fixed_l1=0, fixed_l2=0)
    from src.models.preprocess import DreamChallengePreprocessor2
    preprocess = DreamChallengePreprocessor2(use_rna_seq_track=False)
    from src.models.optimizer import RMSprop, Nesterov, SGD
    optimizer = SGD(nu=0.001, conditioner=RMSprop(), momentum=Nesterov(), minibatch_size=64
                    , convergence_sample=1000)
    from src.models.binding_model import BindingModel, Regularizer
    regularizer = Regularizer(l2=1e-5) #, l1=1e-7)
    model = BindingModel(predictor, optimizer=optimizer, preprocessor=preprocess, dtype=dtype)
                        #, class_weights='balanced')
    return model

def model_constructor_config(config={}, dtype='float32'):
    #### feels bad man
    model = model_constructor(dtype=dtype)
    config['predictor'] = model.predictor.to_config()
    config['optimizer'] = model.optimizer.to_config()
    config['regularizer'] = model.regularizer.to_config()
    config['preprocessor'] = model.preprocessor.to_config()
    config['dtype'] = model.dtype

def main(device='cpu', dtype='float32'):
    import glob
    tracks = glob.glob('data/processed/features/hg19.DNAse.NearestGene5p.*')
    #fixed = 'data/processed/features/MeanTPM.combined.TF.cofactor.txt'
    fixed = None
    labels = ['data/processed/labels/E2F1.train.all.merged.labels.pi.gz'
             , 'data/processed/labels/CTCF.train.all.merged.labels.pi.gz'
             , 'data/processed/labels/MYC.train.all.merged.labels.pi.gz'
             , 'data/processed/labels/GATA3.train.all.merged.labels.pi.gz'
             ]

    import src.config as cfg
    path = os.path.basename(__file__)
    if path.endswith('.py'):
        path = path[:-3]
    path_prefix = os.path.join('results', path)
    path_prefix = os.path.join(path_prefix, cfg.datetime()) + os.sep
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)

    ##open the config
    config_path = path_prefix+'config'
    with cfg.Config.open(config_path) as config:
        config['cross_validate'] = {}
        config = config['cross_validate']
        config['data'] = {}
        parts = list(make_partitions(tracks, fixed, labels, 64, 32, config=config['data']))

        constructor = ModelConstructor(device=device, dtype=dtype)
        config['model'] = {}
        constructor.to_config(config=config['model'])

        nsamples = 8
        sample_size = 2048*8*2
        config['summary'] = {}
        config['summary']['nsamples'] = nsamples
        config['summary']['sample_size'] = sample_size

        from src.scripts.model.cross_validate import cross_validate
        return cross_validate(path_prefix, constructor, parts, nsamples=nsamples, sample_size=sample_size), path_prefix

if __name__ == '__main__':
    tasks, path_prefix = main()
    with open(path_prefix+'summary.txt', 'w') as f:
        first = True 
        for task in tasks:
            df = task()
            df.to_csv(f, sep='\t', float_format='%.5f', header=first, index=False)




