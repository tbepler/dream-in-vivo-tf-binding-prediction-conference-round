import sys
import os

curdir = os.path.dirname(__file__)
rootdir = os.path.dirname(curdir)
sys.path.insert(0, rootdir)
sys.path.insert(1, os.path.join(rootdir, 'rnn'))

if __name__ == '__main__':
    from resid_conv_logit_01_cross_validate import main
    from src.lsf import LSF    
    jobs, path_prefix = main(device='cpu')
    pool = LSF()
    pool.name = 'resid_conv_logit_01_cross_validate'
    pool.queue = 'mcore'
    pool.ncpus = '12'
    pool.runlimit = '172:00'
    pool.resources = ['mem=1500']
    pool.setup = 'export OMP_NUM_THREADS=12'
    for ident in pool.batch(jobs):
        print(ident)



