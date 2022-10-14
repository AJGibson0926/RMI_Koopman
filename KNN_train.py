# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:05:22 2022

@author: gibson48
"""

#import sys
import os
import sys
import json
import pickle
from Koopman_config import collapse_params
import Koopman_NN as KNN
import numpy as np
import helperfns

# Load settings related to dataset
def load_pkl():
    pkl_fn = './output/80-80-170/RMI_2022_07_28_13_44_45_237101_model.pkl' #'./pickles/80-80-170.pkl' #+ sys.argv[1]
    
    with open(pkl_fn, 'rb') as fn:
        params = pickle.load(fn)

    params = collapse_params(params)
    if not os.path.exists(params['folder_name']):
        os.makedirs(params['folder_name'], exist_ok=True)
        
    return params, pkl_fn

params, pkl_fn = load_pkl()

env = json.loads(sys.argv[1])
print(env)
Koop = KNN.KoopmanNet(params, work_dict=env)
print(Koop.worker.params)
#Koop.load_model(pkl_fn)


# data is num_steps x num_examples x n but load flattened version (matrix instead of tensor)
#data_val = np.loadtxt(('./data/TVT/%s/%s_val_x.csv' % (params['train_data'], params['data_name'])), delimiter=',', dtype=np.float64)

#Koop.train()