# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:55:35 2022

@author: gibson48
"""

#import sys
import os
import numpy as np
import pickle
from Koopman_config import collapse_params
import helperfns
import Koopman_NN as KNN

# Load settings related to dataset

# Open the file in binary mode
#pkl_fn = './output/80-80-170-newformat/RMI_2022_07_26_14_06_59_082828_model.pkl' #+ sys.argv[1]
pkl_fn = './output/80-80-170/RMI_2022_07_28_13_44_45_237101_model.pkl'

with open(pkl_fn, 'rb') as fn:
    params = pickle.load(fn)

params = collapse_params(params)
if not os.path.exists(params['folder_name']):
    os.makedirs(params['folder_name'], exist_ok=True)

Koop = KNN.KoopmanNet(params, 'apply')
Koop.load_model(pkl_fn)

# data is num_steps x num_examples x n but load flattened version (matrix instead of tensor)
X = np.loadtxt(('./data/TVT/%s/%s_val_x.csv' % (params['train_data'], params['data_name'])), delimiter=',', dtype=np.float64)
max_shifts_to_stack = helperfns.num_shifts_in_stack(params)
X_stacked = helperfns.stack_data(X, max_shifts_to_stack, params['len_time'])

IC = X_stacked[-1,0,:]