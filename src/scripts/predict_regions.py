import numpy as np
import pandas as pd

def predict_regions(model, store, F, regions, progress=None, chunk_size=50000, overhang_size=600, batch_size=16):
    total = np.sum((regions[:,2]-regions[:,1])//50 - 3)
    completed = 0
    if progress is not None:
        progress(0, 'Predict')
    for R in regions:
        chrom, start, end = R[0], R[1], R[2]
        ## split the region into chunk_size chunks with overhang_size overhang on each end
        ## and process these chunks in batches of batch_size
        for i in range(start, end, (chunk_size-150)*batch_size):
            n = min(end-150, i+(chunk_size-150)*batch_size)
            ## load the batch from the hdf5
            batch_start = i-overhang_size
            batch_end = i+(chunk_size-150)*batch_size+overhang_size+150
            x = store.select(chrom, 'index >= batch_start & index < batch_end')    
            x_start = x.index.values[0]
            x = x.values.astype(model.dtype)
            mask = np.ones(batch_end-batch_start, dtype=model.dtype)
            if batch_end-batch_start > len(x):
                x_ = np.zeros((batch_end-batch_start, x.shape[-1]), dtype=x.dtype)
                s = x_start - batch_start
                x_[s:s+len(x)] = x
                mask[:s] = 0
                mask[s+len(x):] = 0
                x = x_
            ## make the chunks
            x_chunks = []
            mask_chunks = []
            for j in range(0, (chunk_size-150)*batch_size, chunk_size-150):
                x_chunks.append(x[j:j+chunk_size+2*overhang_size])
                mask_chunks.append(mask[j:j+chunk_size+2*overhang_size])
            x = np.stack(x_chunks)
            mask = np.stack(mask_chunks)
            f = None
            if F is not None:
                f = np.stack([F]*len(x))
            y = model.predict_proba((x, mask, f))
            trim = overhang_size//50
            y = y[:,trim:-trim]
            s = np.arange(i, n, 50)
            e = s + 200
            y = y.flatten()[:len(s)]
            completed += len(s)
            if progress is not None:
                progress(completed/total, 'Predict')
            df = pd.DataFrame()
            df['Start'] = s
            df['End'] = e
            df['Prediction'] = y
            df.insert(0, 'Chrom', chrom)
            yield df
