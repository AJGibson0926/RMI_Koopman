# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:12:02 2022

@author: gibson48
"""

import sys
import copy
import pickle
import training
from Koopman_config import collapse_params

# Load settings related to dataset

# Open the file in binary mode
pkl_fn = 'pickles/' + sys.argv[1]

with open(pkl_fn, 'rb') as fn:
    params = pickle.load(fn)

params = collapse_params(params)

# Train
training.main_exp(copy.deepcopy(params))