import os
import sys

class Task(object):
    def __init__(self, path_prefix, partition, model_constructor, nsamples=8, sample_size=2048*4):
        self.path_prefix = path_prefix
        self.partition = partition
        self.model_constructor = model_constructor
        self.nsamples = nsamples
        self.sample_size = sample_size
        ids = self.partition.ident
        self.path = self.path_prefix + '.'.join(ids.tolist()) + os.sep
        self.name = '.'.join(ids.tolist()) 

    def __call__(self):
        dataconfig, train, validate = self.partition() ## load the data
        ids = self.partition.ident
        dirpath = self.path
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        import src.config as cfg
        with cfg.Config.open(dirpath+'config') as subconfig:
            subconfig['data'] = dataconfig
            model = self.model_constructor() ## this is for initial parameters to be sampled independently
                                             ## each time if desired
            subconfig['model'] = model.to_config()
            from src.scripts.model.fit import fit
            subconfig['fit'] = {}
            with open(dirpath+'fit.progress.txt', 'w') as progress_out:
                with open(dirpath+'fit.report.txt', 'w') as report_out:
                    fit(dirpath, model, train, config=subconfig['fit'], report_out=report_out, progress_out=progress_out)
            from src.scripts.model.summary import summary
            subconfig['summary'] = {}
            nsamples = self.nsamples
            sample_size = self.sample_size
            sampled = nsamples > 0
            with open(dirpath+'summary.progress.txt', 'w') as progress_out:
                df = summary(model, validate, sample=sampled, nsamples=nsamples, sample_size=sample_size, progress_out=progress_out, config=subconfig['summary'])
            i = 0
            for name,value in zip(ids.index.tolist(), ids.tolist()):
                df.insert(i, name, value)
                i += 1
            with open(dirpath+'summary.txt', 'w') as f:
                df.to_csv(f, sep='\t', header=True, index=False)
        return df

def cross_validate(path_prefix, model_constructor, partitions, nsamples=8, sample_size=2048*4):
    for partition in partitions:
        yield Task(path_prefix, partition, model_constructor, nsamples=nsamples, sample_size=sample_size)


