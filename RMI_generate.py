# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 13:46:02 2022

@author: gibson48
"""
import sys
import pickle
import numpy as np
from Koopman_config import config_dynsys, save_params

loc = str(sys.argv[1])
worknum = int(sys.argv[2])
fldr = 'data/TVT/' + loc

# Generate params. Only keep the dynsys portion. NN irrelevant: replace this code.
prms = config_dynsys('liner')#, [80,80,170], [0,2], 0.001, 1)

# Save dynsys params to data.pkl
save_params(fldr, prms)

# Divide the ICs up across the worker pool
numICs = prms['numICs']
perworker = int(np.ceil(numICs/worknum))
lastbatch = perworker - (perworker*worknum - numICs)

# First n-1 batches
if worknum > 1:
    A = np.arange(perworker*(worknum - 1))
    A = A.reshape((worknum-1,perworker))
    B = A.tolist()
    
    # Last batch
    L = np.arange(lastbatch) + 1 + A[-1,-1]
    L = L.tolist()
    B.append(L)
else:
    B = [np.arange(numICs).tolist()]

for j in np.arange(worknum):
    work_fn = fldr + '/worker%s.pkl' % j
    
    work_prms = {}
    with open(work_fn, 'wb') as dn:
        work_prms['ICrange'] = B[j]
        pickle.dump(work_prms, dn)