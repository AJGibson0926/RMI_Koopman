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
import control as ctrl
from scipy.integrate import odeint

def grab_KLQR_mats(nn, x, B):
    prms = nn.params
    r = prms['ROM_dim']
    Ir = np.identity(r)
    Qpsi = 10e4*Ir
    R = 10e-4
    
    Lambda = nn.K(x, 'apply')
    Bpsi = nn.Psigrad(x, 'apply') * B
    
    return Lambda, Bpsi, Qpsi, R


def dzdt(z, t, incT, B=0, nn='', zref=0):
    N = int(len(z)/3)
    
    x = z[0:N]
    y = z[N:2*N]
    Gamma = z[2*N:3*N]

    dxdt = np.zeros(N)
    dydt = np.zeros(N)

    #TODO: check this
    # Grab control Bpsi, u
    if nn !='':
        L, Bpsi, Q, R = grab_KLQR_mats(nn, z, B)
        Psi_err = nn.encoder.apply(z) - nn.encoder.apply(zref)
        
        C = ctrl.lqr(L, Bpsi, Q, R)
        u = -C*Psi_err
    else:
        B = 0
        u = 0

    #TODO: Controlled Gammas or uncontrolled?
    for alpha in range(N):
       if incT[alpha]<= t:
          for beta in range(N):
             if alpha != beta:
                lalphabetasq = (x[alpha]-x[beta])**2. + (y[alpha]-y[beta])**2.
                dxdt[alpha] = dxdt[alpha] - 1./(2.*np.pi)*Gamma[beta]*(y[alpha]-y[beta])/lalphabetasq
                dydt[alpha] = dydt[alpha] + 1./(2.*np.pi)*Gamma[beta]*(x[alpha]-x[beta])/lalphabetasq
       
    dzdt = np.concatenate((dxdt,dydt,0*Gamma)) + B*u
    return dzdt



# Load settings related to dataset

# Open the file in binary mode
#pkl_fn = './output/80-80-170-newformat/RMI_2022_07_26_14_06_59_082828_model.pkl' #+ sys.argv[1]
import os
def load_pkl():
    pkl_fn = './output/80-80-170/RMI_2022_07_28_13_44_45_237101_model.pkl'
    
    with open(pkl_fn, 'rb') as fn:
        params = pickle.load(fn)

    params = collapse_params(params)
    if not os.path.exists(params['folder_name']):
        os.makedirs(params['folder_name'], exist_ok=True)
        
    return params, pkl_fn

params, pkl_fn = load_pkl()

Koop = KNN.KoopmanNet(params)
Koop.load_model(pkl_fn)

# data is num_steps x num_examples x n but load flattened version (matrix instead of tensor)
X = np.loadtxt(('./data/TVT/%s/%s_val_x.csv' % (params['train_data'], params['data_name'])), delimiter=',', dtype=np.float64)
max_shifts_to_stack = helperfns.num_shifts_in_stack(params)
X_stacked = helperfns.stack_data(X, max_shifts_to_stack, params['len_time'])

traj = 0
z0 = X_stacked[0,traj,:]

n = int(len(z0)/3) # numvort
B = np.zeros((3*n,3*n))
ID = np.identity(n)
B[2*n:3*n, 2*n:3*n] = ID

prms = Koop.params
t = prms['t']
incT = prms['incT']

###### Solve vortex ODEs ###################
sol = odeint(dzdt, z0, t, args=(incT, B, Koop))