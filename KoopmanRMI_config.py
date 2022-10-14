# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 09:33:28 2022

@author: gibson48
"""

import os
import sys
import pickle
import Koopman_config as Kcnfg

########## MAIN COMPUTE ##########
CHECK_FOLDER = os.path.isdir('pickles')
if not CHECK_FOLDER:
    os.makedirs('pickles')

[data_set, widths, lr, keigfn, tm] = Kcnfg.extract_prms(sys.argv[1:])

# Load the dynsys params from the dataset and formulate the NN
dynsys_params = Kcnfg.load_params([data_set], '')
NN_params = Kcnfg.config_NN(dynsys_params[0], widths, keigfn, lr, tm)

with open('pickles/%s.pkl' %NN_params['id'], 'wb') as fn:
    pickle.dump(NN_params, fn)