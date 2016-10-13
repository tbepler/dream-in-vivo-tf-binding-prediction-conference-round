from __future__ import print_function
import numpy as np
import pandas as pd
import sys
import six

def to_one_hot(x, n=None, dtype=np.float32):
    if n is None:
        n = x.max()
    one_hot = np.zeros((len(x),n), dtype=dtype)
    one_hot[np.arange(len(x)), x] = 1
    return one_hot


def featurize(df, dtype=np.float32):
    dfs = {}
    for colname in df.columns:
        col = df[colname].values
        if 'int' in col.dtype:
            one_hot = to_one_hot(col)
            dfs[col] = pd.DataFrame(one_hot)
        else:
            dfs[col] = pd.DataFrame(col)
    return pd.concat(dfs)


bases_upper = b'ACGTN'
bases_lower = b'acgtn'
base_mapping = {bases[i]:i for bases in [bases_upper, bases_lower] for i in range(len(bases))}

def encode_dna(s):
    return np.fromiter((base_mapping[b] for b in s), dtype=np.int8, count=len(s))

def featurize_dna(s, dtype=np.float32):
    s_encoded = encode_dna(s)
    bases = bases_upper.decode()
    df = pd.DataFrame(to_one_hot(s_encoded, n=len(bases_upper), dtype=dtype), columns=list(bases))
    return df

def featurize_gene_expression(gene_expression_intervals, n):
    plus_strand, minus_strand = expand_gene_expression_intervals(gene_expression_intervals, n)
    df = pd.DataFrame({'+':plus_strand, '-':minus_strand})
    return df

def combine_inputs(genome, dnase, gene_expression_intervals):
    included_chroms = set(dnase.chroms())
    for chrom, s in genome:
        if chrom in included_chroms:
            print('Processing:', chrom, file=sys.stderr)
            dfs = {}
            dfs['DNA'] = featurize_dna(s, dtype=np.float32)
            dfs['DNAse'] = pd.DataFrame({'':np.array(dnase.values(chrom, 0, -1), dtype=np.float32)})
            if gene_expression_intervals is not None:
                dfs['Expression'] = featurize_gene_expression(gene_expression_intervals[chrom], len(s))
            df = pd.concat(dfs, axis=1)
            yield chrom, df

def combine_gene_expression(gene_expression):
    d = {}
    for gene_id, tpm in gene_expression:
        x = d.get(gene_id, [])
        x.append(tpm)
        d[gene_id] = x
    for k,v in six.iteritems(d):
        d[k] = np.mean(v)
    return d

def gene_expression_intervals(expr_dict, gene_annotations):
    expr_chrom_dict = {}
    for chrom, start, end, strand, gene_id in gene_annotations:
        if gene_id in expr_dict:
            expr = expr_dict[gene_id]
            lst = expr_chrom_dict.get(chrom, [])
            lst.append((start, end, strand, expr))
            expr_chrom_dict[chrom] = lst
    return expr_chrom_dict

def expand_gene_expression_intervals(intervals, n):
    plus_strand = np.zeros(n, dtype=np.float32)
    minus_strand = np.zeros(n, dtype=np.float32)
    for start, end, strand, expr in intervals:
        if strand == b'+':
            plus_strand[start:end] = expr
        elif strand == b'-':
            minus_strand[start:end] = expr
        else:
            raise "Strand error: {}, {}, {}, {}".format(start, end, strand, expr)
    return plus_strand, minus_strand 

labels = b'UBA'
label_mapping = {labels[0]:0, labels[1]:1, labels[2]:0}
mask_mapping = {labels[0]:1, labels[1]:1, labels[2]:0}

def encode_labels(L):
    Y = np.fromiter((label_mapping[l] for l in L), dtype=np.int8, count=len(L))
    M = np.fromiter((mask_mapping[l] for l in L), dtype=np.int8, count=len(L))
    return Y, M

def process_labels(L):
    L = [b''.join(l) for l in L]
    return [encode_labels(l) for l in L]

def merge_labels(L):
    cur_chrom = None
    cur_start = 0
    cur_end = 0
    cur_values = None
    for chrom, start, end, values in L:
        if chrom != cur_chrom or end != cur_end+50:
            if cur_chrom is not None:
                yield cur_chrom, cur_start, cur_end, process_labels(cur_values)
            cur_chrom = chrom
            cur_start = start
            cur_end = end
            cur_values = [[v] for v in values]
        else:
            cur_end = end
            for v, vs in zip(values, cur_values):
                vs.append(v)
    if cur_chrom is not None:
        yield cur_chrom, cur_start, cur_end, process_labels(cur_values)

def split_region(start, end, labels, mask, n):
    m = (end-start-150)//50
    for i in range(0, m, n):
        r_end = min(end, start+(i+n+3)*50)
        yield start+i*50, r_end, labels[i:i+n], mask[i:i+n]

def pad_region(start, end, labels, mask, n):
    pad_labels = np.zeros(len(labels)+2*n, dtype=labels.dtype)
    pad_mask = np.zeros(len(mask)+2*n, dtype=mask.dtype)
    pad_labels[n:-n] = labels
    pad_mask[n:-n] = mask
    return start-n*50, end+n*50, pad_labels, pad_mask

def select_region(tracks, start, end):
    n = len(tracks)
    m = end-start
    M = np.ones(m, dtype=np.int8)
    if end > n or start < 0:
        i = 0
        if start < 0:
            i = -start
            start = 0
        M[:i] = 0
        j = m
        if end > n:
            j = n - end
            end = n
        M[j:] = 0
        X = np.zeros((m,tracks.shape[1]), dtype=tracks.dtype)
        X[i:j] = tracks[start:end]
    else:
        X = tracks[start:end]
    return X, M
        













