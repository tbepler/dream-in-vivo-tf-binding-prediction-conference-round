import os
import sys

class Task(object):
    def __init__(self, model_path, TF, predict, dirname, device, dtype, macrobatch_size):
        self.model_path = model_path
        self.TF = TF
        self.predict = predict
        self.dirname = dirname
        self.device = device
        self.dtype = dtype
        self.macrobatch_size = macrobatch_size
        self.path = self.dirname + os.sep
        self.name = TF

    def __call__(self):
        import src.config as cfg
        config_path = os.path.join(self.dirname, 'config')
        with cfg.Config.open(config_path) as config:
            from src.progress import Progress
            config['model_path'] = self.model_path
            config['TF'] = self.TF
            config['predict'] = self.predict
            ## open file for tracking progress
            progress_path = os.path.join(self.dirname, 'progress.txt')
            with open(progress_path, 'w') as f:
                progress = Progress(out=f)
                ## initialize the device
                if self.device is not None:
                    print('[In Progress] initialize device: {}'.format(self.device), file=progress)
                    cfg.set_device(self.device)
                    print('[Done] initialize device: {}'.format(self.device), file=progress)
                ## open the model file
                import importlib.machinery
                loader = importlib.machinery.SourceFileLoader('model_src', self.model_path)
                mod = loader.load_module()
                ## load the training data
                print('[In Progress] Load data', file=progress)
                config['data'] = {}
                train_data = mod.load_data(self.TF, config=config['data'])
                print('[Done] Load data', file=progress)
                ## load the model
                print('[In Progress] Load model', file=progress)
                kwargs = {}
                if self.dtype is not None:
                    kwargs['dtype'] = self.dtype
                model = mod.load_model(**kwargs)
                config['model'] = model.to_config()
                print('[Done] Load model', file=progress)
                ## fit the model
                from src.scripts.model.fit import fit
                print('[In Progress] Fit model', file=progress)
                config['fit'] = {}
                dirpath = self.path
                report_path = os.path.join(self.dirname, 'fit.report.txt')
                kwargs = {}
                kwargs['config'] = config['fit']
                kwargs['progress'] = progress
                kwargs['macrobatch_size'] = self.macrobatch_size
                with open(report_path, 'w') as report_out:
                    kwargs['report_out'] = report_out
                    fit(dirpath, model, train_data, **kwargs)
                print('[Done] Fit model', file=progress)
                ## compute the leaderboard predictions if called for
                if 'leaderboard' in self.predict:
                    from src.scripts.predict_leaderboard import predict_leaderboard
                    predict_leaderboard(self.TF, model, mod.tracks, mod.fixed, self.dirname, progress=progress)
                ## same for final predictions
                if 'final' in self.predict:
                    from src.scripts.predict_final import predict_final
                    predict_final(self.TF, model, mod.tracks, mod.fixed, self.dirname, progress=progress)

def fit_production_model(model_id, TFs, predict, destdir, device, dtype, macrobatch_size):
    import src.config as cfg
    if model_id.endswith('.py'):
        model_path = model_id
        model_id = os.path.basename(model_id)[:-3]
    else:
        model_path = os.path.join('models', model_id + '.py')
    if 'all' in TFs:
        from src.dataset import reference
        TFs = reference['TF Name'].tolist()
    for TF in TFs:
        dirname = os.path.join(destdir, TF, model_id, cfg.datetime())
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        yield Task(model_path, TF, predict, dirname, device, dtype, macrobatch_size)
