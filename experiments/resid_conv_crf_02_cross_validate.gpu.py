import sys
import os

curdir = os.path.dirname(__file__)
rootdir = os.path.dirname(curdir)
sys.path.insert(0, rootdir)
sys.path.insert(1, os.path.join(rootdir, 'rnn'))

if __name__ == '__main__':
    from resid_conv_crf_02_cross_validate import main
    from src.lsf import LSF    
    jobs, path_prefix = main(device='gpu')
    pool = LSF()
    pool.name = 'resid_conv_crf_02_cross_validate'
    pool.queue = 'gpu'
    pool.runlimit = '36:00'
    pool.resources = ['mem=16000', 'ngpus=1']
    for ident in pool.batch(jobs):
        print(ident)



