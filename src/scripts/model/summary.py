import numpy as np
import pandas as pd
import sys
import random

from src.progress import Progress
from src.models.optimizer import blocks

def cross_entropy(y_true, y_pred):
    return np.mean(-y_true*np.log(y_pred) - (1-y_true)*np.log(1-y_pred))

def average_positive_prediction(y_true, y_pred):
    return y_pred[y_true==1].mean()

def average_negative_prediction(y_true, y_pred):
    return np.mean(1-y_pred[y_true==0])

def calc_stats(model, data, progress_out):
    y_true = []
    y_pred = []
    progress = Progress(out=progress_out)
    pbar = progress.progress_monitor()
    pbar(0, 'Predict')
    n = len(data)
    prog = 0
    for b, block in blocks(data, 512):
        X, Y = block
        Y_M = Y[...,1]
        Y = Y[...,0]
        P = model.predict_proba(X)
        y_true.extend(Y[Y_M==1].flatten().tolist())
        y_pred.extend(P[Y_M==1].flatten().tolist())
        prog += b
        pbar(prog/n, 'Predict')

    from sklearn.metrics import average_precision_score, roc_auc_score
    df = pd.DataFrame()
    df['auPRC'] = average_precision_score(y_true, y_pred)
    df['auROC'] = roc_auc_score(y_true, y_pred)
    df['Cross Entropy'] = cross_entropy(y_true, y_pred)
    df['AvgPP'] = average_positive_prediction(y_true, y_pred)
    df['AvgNP'] = average_negative_prediction(y_true, y_pred)
    df['Support'] = y_true.sum()
    df['Total'] = len(y_true)
    return df

def num_examples(x):
    if x is None:
        return 0 #err... wot!
    if type(x) == tuple:
        return max(num_examples(z) for z in x)
    return x.shape[0]

def sample_stats(model, data, sample_size, num_samples, progress_out, chunk_size):
    y_true = []
    y_pred = []
    progress = Progress(out=progress_out)
    pbar = progress.progress_monitor()
    pbar(0, 'Predict')
    chunks_per_sample = (sample_size+chunk_size//2)//chunk_size
    n = chunks_per_sample*num_samples
    prog = 0
    for _ in range(num_samples):
        yt, yp = [], []
        for _ in range(chunks_per_sample):
            block = next(data)
            b = num_examples(block)
            X, Y = block
            Y_M = Y[...,1]
            Y = Y[...,0]
            P = model.predict_proba(X)
            yt.append(Y[Y_M==1].flatten())
            yp.append(P[Y_M==1].flatten())
            prog += 1
            pbar(prog/n, 'Predict')
        yt = np.concatenate(yt)
        yp = np.concatenate(yp)
        y_true.append(yt)
        y_pred.append(yp)

    from sklearn.metrics import average_precision_score, roc_auc_score
    auprc = [average_precision_score(yt, yp) for yt,yp in zip(y_true, y_pred)]
    auroc = [roc_auc_score(yt, yp) for yt,yp in zip(y_true, y_pred)]
    cent = [cross_entropy(yt, yp) for yt,yp in zip(y_true, y_pred)]
    avg_pp = [average_positive_prediction(yt, yp) for yt,yp in zip(y_true, y_pred)]
    avg_np = [average_negative_prediction(yt, yp) for yt,yp in zip(y_true, y_pred)]
    support = [yt.sum() for yt in y_true]
    total = [len(yt) for yt in y_true]
    df = pd.DataFrame()
    df['Sample'] = np.arange(num_samples)
    df['auPRC'] = auprc
    df['auROC'] = auroc
    df['Cross Entropy'] = cent
    df['AvgPP'] = avg_pp
    df['AvgNP'] = avg_np
    df['Support'] = support
    df['Total'] = total
    df_stats = pd.DataFrame()
    df_stats['Sample'] = ['Mean', 'Median', 'Stdev']
    df_stats['auPRC'] = [np.mean(auprc), np.median(auprc), np.sqrt(np.var(auprc))]
    df_stats['auROC'] = [np.mean(auroc), np.median(auroc), np.sqrt(np.var(auroc))]
    df_stats['Cross Entropy'] = [np.mean(cent), np.median(cent), np.sqrt(np.var(cent))]
    df_stats['AvgPP'] = [np.mean(avg_pp), np.median(avg_pp), np.sqrt(np.var(avg_pp))]
    df_stats['AvgNP'] = [np.mean(avg_np), np.median(avg_np), np.sqrt(np.var(avg_np))]
    df_stats['Support'] = [np.mean(support), np.median(support), np.sqrt(np.var(support))]
    df_stats['Total'] = [np.mean(total), np.median(total), np.sqrt(np.var(total))]
    df = df.append(df_stats)
    return df

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

def summary(model, data, sample=False, nsamples=10, sample_size=4096*2, macrobatch_size=512, progress_out=sys.stderr, config={}):
    if sample:
        config['sampled'] = True
        config['nsamples'] = nsamples
        config['sample_size'] = sample_size
        macrobatch_size = min(macrobatch_size, sample_size)
        data = make_sampler(data, config=config, macrobatch_size=macrobatch_size)
        df = sample_stats(model, data, sample_size, nsamples, progress_out, macrobatch_size)
    else:
        config['sampled'] = False
        df = calc_stats(model, data, progress_out)
    return df
