#!/usr/bin/env python

if __name__=='__main__':
    import sys
    import os
    root_split = os.path.dirname(os.path.abspath(__file__)).split('/')[:-2]
    sys.path.append('/'.join(root_split))
    sys.path.append('/'.join(root_split+['rnn']))
    import pickle
    with open(sys.argv[1], 'rb') as f:
        thunk = pickle.load(f)
    thunk()

