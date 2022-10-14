# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:49:03 2022

@author: gibson48
"""

## Builds the hyperparameter grid to search through for hyperparameter tuning

import numpy as np

psi_widths_grid = np.array([[2, 10], [80, 160]]) # Between 2 and 10 layers deep, 50 and 160 nodes wide
lambda_widths_grid = np.array([[1,1], [170, 300]]) # Between 1 and 3 layers deep, 20 and 130 nodes wide

# Cycle through all the possible combinations of psi and lambda network sizes. Generate a pkl file for each one: i.e. give each combination its own line in the hyperparameter file.
P = psi_widths_grid[:,1] - psi_widths_grid[:,0]
L = lambda_widths_grid[:,1] - lambda_widths_grid[:,0]

A = range(psi_widths_grid[0,0],psi_widths_grid[0,1]+1)
B = range(psi_widths_grid[1,0],psi_widths_grid[1,1]+1)
C = range(lambda_widths_grid[0,0],lambda_widths_grid[0,1]+1)
D = range(lambda_widths_grid[1,0],lambda_widths_grid[1,1]+1)

l = []

for a in A:
    for b in B:
        for c in C:
            for d in D:
                l.append([a,b,c,d])

np.savetxt('RMI_hyperparams.csv', l, delimiter=',')