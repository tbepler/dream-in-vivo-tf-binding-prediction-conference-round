import pandas as pd
import numpy as np
import subprocess
import os
import shutil

from src.scripts.predict_regions import predict_regions

def get_file(paths, content):
    for path in paths:
        if content in path:
            return path
    return None

def cross_entropy(y_true, y_pred):
    return np.mean(-y_true*np.log(y_pred) - (1-y_true)*np.log(1-y_pred))

def average_positive_prediction(y_true, y_pred):
    return y_pred[y_true==1].mean()

def average_negative_prediction(y_true, y_pred):
    return np.mean(1-y_pred[y_true==0])

def make_summary(y_true, y_pred):
    from sklearn.metrics import average_precision_score, roc_auc_score
    df = pd.DataFrame()
    df['auPRC'] = [average_precision_score(y_true, y_pred)]
    df['auROC'] = [roc_auc_score(y_true, y_pred)]
    df['Cross Entropy'] = [cross_entropy(y_true, y_pred)]
    df['AvgPP'] = [average_positive_prediction(y_true, y_pred)]
    df['AvgNP'] = [average_negative_prediction(y_true, y_pred)]
    df['Support'] = [y_true.sum()]
    df['Total'] = [len(y_true)]
    return df

def summary_training(TF, model, tracks, fixed, destdir, progress=None):
    ## get the leaderboard cell types from the reference table
    from src.dataset import reference
    lb_types = reference['Training Cell Types'][reference['TF Name']==TF].values[0]
    if lb_types == '':
        return # there are no leaderboard cell lines for this TF
    lb_types = lb_types.split(', ')
    ## check for this IMR-90 nonsense
    try:
        i = lb_types.index('IMR-90')
        lb_types[i] = 'IMR90'
    except ValueError:
        pass
    ## load the training regions
    regions = pd.read_csv('data/raw/annotations/train_regions.blacklistfiltered.merged.bed', sep='\t', header=None).values
    labels_path = 'data/raw/ChIPseq/labels/{}.train.labels.tsv.gz'.format(TF)
    ## load the fixed data
    if fixed is not None:
        fixed = pd.read_csv(fixed, sep='\t', index_col=0)[lb_types]
    ## generate the summary for each leaderboard cell type
    with open(os.path.join(destdir, 'T.summary.txt'), 'w') as outfile:
        first = True
        for cell_line in lb_types:
            if progress is not None:
                print('[In progress] Predict {}'.format(cell_line), file=progress)
            F = None
            if fixed is not None:
                F = fixed[cell_line].values.astype(model.dtype)
            ## open the hdf5 tracks
            store = pd.HDFStore(get_file(tracks, cell_line), mode='r')
            ## reader for getting labels in chunks
            labelsreader = pd.read_csv(labels_path, sep='\t', iterator=True)
            ## calculate the predictions
            yh = []
            y = []
            n = 0
            callback = None if progress is None else progress.progress_monitor()
            for df in predict_regions(model, store, F, regions, progress=callback):
                labels = labelsreader.get_chunk(len(df))
                ## compare to make sure regions are the same
                lr = labels[['chr', 'start', 'stop']].values
                dr = df[['Chrom', 'Start', 'End']].values
                I = (lr != dr).any(1)
                if I.any():
                    lines = np.arange(n, n+len(df))[I.values]
                    raise Exception('Region mistmatch: lines = {}'.format(lines))
                n += len(df)
                ## process the labels to remove ambiguous regions and convert to 0/1
                dy = df['Prediction'].values
                ly = labels[cell_line].values
                mask = (ly != 'A')
                dy = dy[mask]
                ly = ly[mask]
                positive = (ly == 'B')
                ly = np.zeros(ly.shape, dtype=np.int8)
                ly[positive] = 1
                ## record
                yh.append(dy)
                y.append(ly)
            ## generate the classification report
            yh = np.concatenate(yh, axis=0)
            y = np.concatenate(y, axis=0)
            df = make_summary(y, yh)
            df.insert(0, 'Cell Line', cell_line)
            df.to_csv(outfile, sep='\t', header=first, index=False)
            outfile.flush()
            first = False
            ## close the hdf5
            store.close()
            if progress is not None:
                print('[Done] Predict {}'.format(cell_line), file=progress)

