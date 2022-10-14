# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:50:37 2022

@author: gibson48
"""

from numpy import random

N = 320
lr_list = [0.0001]

for k in range(len(lr_list)):
    for d in range(N):
        runfile('KoopmanRMI_config.py', args='%s %s' % (random.randint(0,94000), lr_list[k]))