from __future__ import print_function

import itertools
import six
import os
import sys
import pyBigWig
import pandas as pd
import numpy as np
import gzip

import src.data as data
import src.parser as parse

def read_gene_expression(path):
    with open(path) as f:
        for x in parse.read_gene_expression(f):
            yield x

def load_gene_expression(paths):
    expr = [read_gene_expression(path) for path in paths]
    expr = data.combine_gene_expression(itertools.chain(*expr))
    return expr

def load_gene_5primes(path):
    d = {}
    with gzip.open(path) as f:
        for chrom, start, end, strand, gene_id in parse.read_gene_annotations(f):
            if strand == b'+':
                pos = (chrom, start-1)
            elif strand == b'-':
                pos = (chrom, end)
            d[gene_id] = pos
    return d


def read_chrom_sizes(path):
    d = {}
    with open(path) as f:
        for line in f:
            if line != '':
                tokens = line.split()
                d[tokens[0]] = int(tokens[1])
    return d

def nearest_gene_5p_track(chrom_sizes_path, gene_annotations_path, rna_seq_paths):
    chrom_sizes = read_chrom_sizes(chrom_sizes_path)
    gene_5primes = load_gene_5primes(gene_annotations_path)
    gene_expression = load_gene_expression(rna_seq_paths)

    exp_5p = []
    for gene_id, tpm in six.iteritems(gene_expression):
        if gene_id in gene_5primes:
            chrom, pos = gene_5primes[gene_id]
            exp_5p.append((chrom, pos, tpm))
    exp_5p.sort()
    chroms = list(zip(*exp_5p))[0]
    chroms = sorted(list(set(chroms)))
    start = 0
    for i in range(len(exp_5p)):
        chrom, pos, tpm = exp_5p[i]
        if i+1 == len(exp_5p) or exp_5p[i+1][0] != chrom:
            end = chrom_sizes[chrom]
            nstart = 0
        else:
            _, n, _ = exp_5p[i+1]
            end = pos + (n-pos)//2 + 1
            nstart = end
        yield chrom, start, end, tpm
        start = nstart

def track_id(path):
    name = os.path.basename(path)
    return name.split('.')[0]


class Intervals(object):
    def __init__(self, d):
        self.d = d
        self.chroms = d.keys()

    def values(self, chrom, start, end):
        n = end-start
        array = np.zeros(n, dtype=np.float32)
        if chrom in self.chroms:
            ints = self.d[chrom]
            for start, end, v in ints:
                array[start:end] = v
        return array

def read_bed(f):
    d = {}
    for line in f:
        if line != '':
            tokens = line.split()
            chrom = tokens[0].decode()
            start = int(tokens[1])
            end = int(tokens[2])
            v = float(tokens[3])
            l = d.get(chrom, [])
            l.append((start, end, v))
            d[chrom] = l
    return Intervals(d)

class BigWig(object):
    def __init__(self, bw):
        self.bw = bw
        self.chroms = bw.chroms().keys()

    def values(self, chrom, start, end):
        array = np.zeros(end-start, dtype=np.float32)
        if chrom in self.chroms:
            ints = self.bw.intervals(chrom)
            for start, end, v in ints:
                array[start:end] = v
        return array

def open_track(path):
    if path.endswith('.bed.gz'):
        with gzip.open(path) as f:
            return track_id(path), read_bed(f)
    elif path.endswith('.bigwig'):
        return track_id(path), BigWig(pyBigWig.open(path))
    else:
        raise NotImplementedError
        
def open_tracks(paths):
    return sorted([open_track(path) for path in paths])

def chroms_set(tracks):
    s = set()
    for _, track in tracks:
        s.update(track.chroms)
    return s

def combine_tracks(genome, tracks, output):
    store = pd.HDFStore(output, 'w', complevel=9, complib='blosc')
    tracks = open_tracks(tracks)    
    chroms = chroms_set(tracks)
    with gzip.open(genome) as f:
        for chrom, s in parse.read_fasta(f):
            if chrom in chroms:
                print('# Processing:', chrom)
                sys.stdout.flush()
                df = data.featurize_dna(s)
                for track_id, bw in tracks:
                    df[track_id] = bw.values(chrom, 0, len(s))
                store.put(chrom, df, format='table')

