from __future__ import print_function
 
import sys
import gzip
import pickle
import pandas as pd

import src.data as data

def unpickle_iter(f):
    while True:
        try:
            yield pickle.load(f)
        except EOFError:
            break

def unpickle_features(f):
    for chrom, chunks in unpickle_iter(f):
        df = pd.concat(chunks, axis=0)
        yield chrom, df

def load_features(regions, file_paths):
    cell_lines = set(regions['Cell Line'])
    for cell_line in cell_lines:
        cell_line_regions = regions.loc[regions['Cell Line']==cell_line]
        paths = [path for path in file_paths if cell_line in path]
        for path in paths:
            print('Reading:', path, file=sys.stderr)
            with gzip.open(path) as f:
                for chrom, features in unpickle_features(f):
                    features = features.values
                    chrom_regions = cell_line_regions.loc[cell_line_regions['Chrom']==chrom]
                    for _,row in chrom_regions.iterrows():
                        region_features, feat_mask = data.select_region(features, row['Start'], row['End'])
                        yield region_features, feat_mask, row

def read_fasta(f):
    name = None
    x = []
    for line in f:
        line = line.strip()
        if line != b'':
            if line.startswith(b'>'):
                if name is not None:
                    yield name, b''.join(x)
                name = line[1:].decode()
                x = []
            else:
                x.append(line.upper())
    if name is not None:
        yield name, b''.join(x)

def parse_gff3_flags(s):
    tokens = s.split(b';')
    d = {}
    for token in tokens:
        k,v = token.split(b'=')
        d[k] = v.decode()
    return d

def read_gene_annotations(f):
    for line in f:
        line = line.strip()
        if line != b'' and not line.startswith(b'#'):
            tokens = line.split()
            ident = tokens[2]
            if ident == b'gene':
                chrom = tokens[0].decode()
                start = int(tokens[3])
                end = int(tokens[4])
                strand = tokens[6]
                flags = parse_gff3_flags(tokens[8])
                yield chrom, start, end, strand, flags[b'gene_id']

def read_gene_expression(f):
    next(f) #discard header
    for line in f:
        line = line.strip()
        if line != b'':
            tokens = line.split()
            gene_id = tokens[0]
            tpm = float(tokens[5])
            yield gene_id, tpm

def read_regions(f, header=False):
    if header:
        next(f) #discard the header
    for line in f:
        line = line.strip()
        if line != b'':
            tokens = line.split()
            chrom = tokens[0].decode()
            start = int(tokens[1])
            end = int(tokens[2])
            values = tokens[3:]
            if len(values) > 0:
                yield chrom, start, end, values
            else:
                yield chrom, start, end

def read_labels(f):
    header = next(f)
    cell_lines = header.decode().split()[3:]
    return cell_lines, read_regions(f, header=False)


