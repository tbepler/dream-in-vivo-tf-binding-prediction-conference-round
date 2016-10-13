import pandas as pd
import numpy as np
import subprocess
import os

from src.scripts.predict_regions import predict_regions

def get_file(paths, content):
    for path in paths:
        if content in path:
            return path
    return None

def predict_final(TF, model, tracks, fixed, destdir, progress=None):
    ## get the final round cell types from the reference table
    from src.dataset import reference
    cell_types = reference['Final Submission Cell Types'][reference['TF Name']==TF].values[0]
    if cell_types == '':
        return #there are no final cell lines for this TF
    cell_types = cell_types.split(', ')
    ## load the final round test regions
    regions = pd.read_csv('data/raw/annotations/test_regions.blacklistfiltered.merged.bed', sep='\t', header=None).values
    ## load the fixed data
    if fixed is not None:
        fixed = pd.read_csv(fixed, sep='\t', index_col=0)[cell_types]
    ## generate predictions for each final round cell type
    for cell_line in cell_types:
        if progress is not None:
            print('[In progress] Predict {}'.format(cell_line), file=progress)
        F = None
        if fixed is not None:
            F = fixed[cell_line].values.astype(model.dtype)
        ## open the hdf5 tracks
        store = pd.HDFStore(get_file(tracks, cell_line), mode='r')
        ## open the destination file
        path = '.'.join(['F', TF, cell_line, 'tab.gz'])
        leaderboard_path = os.path.join(destdir, path)
        #with gzip.open(leaderboard_path, 'w') as gz:
        gzip = subprocess.Popen('gzip -c > {}'.format(leaderboard_path), shell=True, stdin=subprocess.PIPE, universal_newlines=True)
        ## write the predictions
        callback = None if progress is None else progress.progress_monitor()
        for df in predict_regions(model, store, F, regions, progress=callback):
            df.to_csv(gzip.stdin, sep='\t', header=False, index=False)
        ## finish writing and close subprocess
        gzip.communicate()
        ## close the hdf5
        store.close()
        if progress is not None:
            print('[Done] Predict {}'.format(cell_line), file=progress)
        

if __name__=='__main__':
    print(reference)
    predict_final('MAFK', None, None, None, None)

