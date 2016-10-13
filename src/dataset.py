import pandas as pd
import numpy as np
import pickle

import blaze ## SOOOOOO this seems to prevent a bug in h5py when indexing HDF5 files...
import h5py

## load the reference table
reference = pd.read_csv('data_reference.txt', sep='\t').fillna('')

def get_cell_lines(TF):
    row = reference.loc[reference['TF Name'] == TF]
    cell_lines = []
    train_cell_lines = row['Training Cell Types'].values[0]
    if train_cell_lines != '':
        cell_lines += [s.strip() for s in train_cell_lines.split(',')]
    val_cell_lines = row['Leaderboard Cell Types'].values[0]
    if val_cell_lines != '':
        cell_lines += [s.strip() for s in val_cell_lines.split(',')]
    test_cell_lines = row['Final Submission Cell Types'].values[0]
    if test_cell_lines != '':
        cell_lines += [s.strip() for s in test_cell_lines.split(',')]
    try:
        i = cell_lines.index('IMR-90')
        cell_lines[i] = 'IMR90'
    except ValueError:
        pass
    return cell_lines

def get_training_cell_lines(TF):
    row = reference.loc[reference['TF Name'] == TF]
    cell_lines = []
    train_cell_lines = row['Training Cell Types'].values[0]
    if train_cell_lines != '':
        cell_lines += [s.strip() for s in train_cell_lines.split(',')]
    try:
        i = cell_lines.index('IMR-90')
        cell_lines[i] = 'IMR90'
    except ValueError:
        pass
    return cell_lines

def load(track_paths, fixed_path, regions_path):
    regions = load_regions(regions_path)
    Y = None
    if type(regions) == tuple:
        regions, Y = regions
    cell_lines = sorted(list(set(regions[:,0])))
    tracks = {k: find_file(track_paths, k) for k in cell_lines}
    tracks = HDF5Tracks(tracks)
    fixed = Fixed(fixed_path)
    dataset = Dataset(regions, tracks, fixed)
    if Y is not None:
        return dataset, Y
    pass

def find_file(paths, cell_line):
    ## boy this is a hack - but there is a bug in the data
    ## where IMR-90 is called IMR90 in file names
    if cell_line == 'IMR-90':
        cell_line = 'IMR90'
    for path in paths:
        if cell_line in path:
            return path

def slice_regions(R, nbins, stride, binwidth=200, binstride=50, pad='fill', Y=None):
    sliced = []
    y_sliced = []
    pad = (binwidth//binstride) - 1
    start = -nbins+1
    end = -pad
    for i in range(len(R)):
        r = R[i]
        binlen = (r[3] - r[2])//binstride
        bin_starts = np.arange(start, binlen+end, stride)
        bin_ends = bin_starts+nbins+pad
        #bin_starts = np.maximum(bin_starts, 0)
        #bin_ends = np.minimum(bin_ends, binlen)
        slices = np.empty((len(bin_starts), 4), dtype=object)
        slices[:,0] = r[0]
        slices[:,1] = r[1]
        slices[:,2] = bin_starts*binstride + r[2]
        slices[:,3] = bin_ends*binstride + r[2]
        if Y is not None:
            y = Y[i]
            y_slices = np.zeros((len(bin_starts), nbins, 2), dtype=y.dtype)
            for j in range(len(bin_starts)):
                s = np.maximum(-bin_starts[j], 0)
                s_ = np.maximum(bin_starts[j], 0)
                e = np.maximum(s_+nbins-s-len(y), 0)
                y_slices[j, s:nbins-e] = y[s_:s_+nbins-s]
            ##remove any slices with only 0 masks
            all_zero_mask = np.all(y_slices[:,:,1] == 0, axis=-1)
            slices = slices[~all_zero_mask]
            y_slices = y_slices[~all_zero_mask]
            y_sliced.append(y_slices)
        sliced.append(slices)
    sliced = np.concatenate(sliced, axis=0)
    if len(y_sliced) > 0:
        y_sliced = np.concatenate(y_sliced, axis=0)
        return sliced, y_sliced
    return sliced

def load_regions(path, dtype=None):
    if path.endswith('.gz'):
        import gzip
        with gzip.open(path) as f:
            array = pickle.load(f).values
    else:
        with open(path, 'rb') as f:
            array = pickle.load(f).values
    ##hack for fixing the IMR-90 vs IMR90 problem
    ## file names contain IMR90 but original label files may contain IMR-90
    I = array[:,0] == 'IMR-90'
    array[I,0] = 'IMR90'
    if array.shape[1] > 4:
        Y = to_matrix(array[:,4])
        if dtype is not None and Y.dtype != dtype:
            Y = Y.astype(dtype)
        mask = to_matrix(array[:,5]).astype(Y.dtype)
        return array[:,:4], np.stack((Y, mask), axis=-1) 
    return array

def to_matrix(v):
    m = len(v)
    n = max(len(x) for x in v)
    A = np.zeros((m,n), dtype=v[0].dtype)
    for i in range(m):
        k = len(v[i])
        A[i,:k] = v[i]
    return A

def load_features(track_paths, fixed_path, regions, dtype=None, hdf='h5py'):
    cell_lines = sorted(list(set(regions[:,0])))
    tracks = {k: find_file(track_paths, k) for k in cell_lines}
    if hdf == 'h5py':
        tracks = H5pyTracks(tracks, dtype=dtype)
    elif hdf == 'pytables':
        tracks = HDF5Tracks(tracks, dtype=dtype)
    else:
        raise Exception('No hdf option: {}'.format(hdf))
    fixed = None
    if fixed_path is not None:
        fixed = Fixed(fixed_path, dtype=dtype)
    return Dataset(regions, tracks, fixed)

class Dataset(object):
    def __init__(self, regions, tracks, fixed):
        self.regions = regions
        self.tracks = tracks
        self.fixed = fixed

    @property
    def shape(self):
        n = 0 if self.fixed is None else self.fixed.shape[0]
        if n > 0:
            m = len(set(self.cell_line))
            n = (m,n)
        return (7,0,n)

    @property
    def fixed_array(self):
        if self.fixed is None:
            return None
        return self.fixed.df.values.T

    @property
    def cell_line(self):
        return self.regions[:,0]

    @property
    def chrom(self):
        return self.regions[:,1]

    @property
    def start(self):
        return self.regions[:,2]

    @property
    def end(self):
        return self.regions[:,3]

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, I):
        R = self.regions[I]
        X, mask = self.tracks[R]
        F = None
        if self.fixed is not None:
            F = self.fixed[R]
        return X, mask, F

    def select(self, I):
        copy = Dataset.__new__(Dataset)
        copy.regions = self.regions[I]
        copy.tracks = self.tracks.select(copy.regions)
        copy.fixed = None
        if self.fixed is not None:
            copy.fixed = self.fixed.select(copy.regions)
        return copy

class H5pyTracks(object):
    def __init__(self, path_dict, dtype=None):
        self.path_dict = path_dict
        self.dtype = dtype
        self.hdfs = {}
        self.open()

    def open(self):
        self.hdfs = {key: h5py.File(path, mode='r') for key,path in self.path_dict.items()}
        return self

    def close(self):
        for _,v in self.hdfs:
            v.close()
        self.hdfs = {}

    def to_config(self):
        return self.path_dict

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['hdfs']
        return state

    def __setstate__(self, s):
        self.__init__(s['path_dict'], dtype=s['dtype'])

    def __getitem__(self, R):
        if R.ndim > 1:
            m = R.shape[0]
            n = np.max(R[:,3]-R[:,2]) ## need to allocate enough space for longest region
            x, mask = self[R[0]]
            X = np.zeros((m,n,x.shape[-1]), dtype=x.dtype)
            M = np.zeros((m,n), dtype=mask.dtype)
            X[0,:len(x)] = x
            M[0,:len(mask)] = mask
            for i in range(1, m):
                x, mask = self[R[i]]
                X[i,:len(x)] = x
                M[i,:len(mask)] = mask
            return X, M
        hdf = self.hdfs[R[0]]
        start = R[2]
        end = R[3]
        chrom = hdf[R[1]]['table']
        s = start
        if s < 0:
            s = 0
        e = end
        if e > len(chrom):
            e = len(chrom)
        x = chrom[s:e]['values_block_0']
        if self.dtype is not None and x.dtype != self.dtype:
            x = x.astype(self.dtype)
        mask = np.ones(len(x), dtype=x.dtype)
        if len(x) < end-start:
            x_start = s-start
            x_end = x_start+len(x)
            x_ = np.zeros((end-start, x.shape[1]), dtype=x.dtype)
            mask_ = np.zeros(end-start, dtype=x.dtype)
            x_[x_start:x_end] = x
            mask_[x_start:x_end] = 1
            x = x_
            mask = mask_
        return x, mask

    def select(self, R):
        cell_lines = set(R[:,0])
        copy = H5pyTracks.__new__(H5pyTracks)
        copy.path_dict = {k: self.path_dict[k] for k in cell_lines if k in self.path_dict}
        copy.dtype = self.dtype
        copy.hdfs = {k: self.hdfs[k] for k in cell_lines if k in self.hdfs}
        return copy

class HDF5Tracks(object):
    def __init__(self, path_dict, dtype=None):
        self.path_dict = path_dict
        self.dtype = dtype
        self.hdfs = {}
        self.open()

    def open(self):
        self.hdfs = {key: pd.HDFStore(path, mode='r') for key,path in self.path_dict.items()}
        return self

    def close(self):
        for _,v in self.hdfs:
            v.close()
        self.hdfs = {}

    def to_config(self):
        return self.path_dict

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['hdfs']
        return state

    def __setstate__(self, s):
        self.__init__(s['path_dict'], dtype=s['dtype'])

    def __getitem__(self, R):
        if R.ndim > 1:
            m = R.shape[0]
            n = np.max(R[:,3]-R[:,2]) ## need to allocate enough space for longest region
            x, mask = self[R[0]]
            X = np.zeros((m,n,x.shape[-1]), dtype=x.dtype)
            M = np.zeros((m,n), dtype=mask.dtype)
            X[0,:len(x)] = x
            M[0,:len(mask)] = mask
            for i in range(1, m):
                x, mask = self[R[i]]
                X[i,:len(x)] = x
                M[i,:len(mask)] = mask
            return X, M
        store = self.hdfs[R[0]]
        start = R[2]
        end = R[3]
        x = store.select(R[1], 'index>=start & index<end')
        x_start = x.index.values[0] - start
        x_end = x_start + x.shape[0]
        x = x.values
        if self.dtype is not None and x.dtype != self.dtype:
            x = x.astype(self.dtype)
        mask = np.ones(x.shape[0], dtype=x.dtype)
        if end-start > x.shape[0]: ### need to pad x and mask to region size
            x_ = np.zeros((end-start, x.shape[1]), dtype=x.dtype)
            mask_ = np.zeros(end-start, dtype=x.dtype)
            x_[x_start:x_end] = x
            mask_[x_start:x_end] = 1
            x = x_
            mask = mask_
        return x, mask

    def select(self, R):
        cell_lines = set(R[:,0])
        copy = HDF5Tracks.__new__(HDF5Tracks)
        copy.path_dict = {k: self.path_dict[k] for k in cell_lines if k in self.path_dict}
        copy.dtype = self.dtype
        copy.hdfs = {k: self.hdfs[k] for k in cell_lines if k in self.hdfs}
        return copy

class Fixed(object):
    def __init__(self, path, dtype=None):
        self.path = path
        self.df = pd.read_csv(path, sep='\t', index_col=0)
        if dtype is not None:
            self.df = self.df.astype(dtype)

    @property
    def shape(self):
        return self.df.shape

    def to_config(self):
        return self.path

    def __getitem__(self, R):
        if R.ndim > 1:
            return self.df[R[:,0]].values.T
        return self.df[R[0]].values

    def select(self, R):
        #cell_lines = sorted(list(set(R[:,0])))
        copy = Fixed.__new__(Fixed)
        copy.path = self.path
        copy.df = self.df
        return copy


