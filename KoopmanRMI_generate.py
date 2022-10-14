# -*- coding: utf-8 -*-

import sys
import pickle
import numpy as np
import control as ctrl
from Koopman_config import save_params
from scipy.integrate import odeint
##########################################################
# Constants
##########################################################
PI = np.pi

# Generate initial condition, i.e. arrangements and Gammas, i.e. positions and strengths of vortices
def arrangeV(prms, ICnum):
    aux = prms['aux']
    
    config = prms['config']
    Gamma = prms['Gamma']
    N = prms['numvort']
    
    if config=='liner' or config=='dblliner':
        slope = aux['slopes'][ICnum]
        linerlength = aux['linearlength']
        deltax = aux['deltax']

        if config=='liner':
            topvorticesx = np.linspace(deltax, linerlength, int(N/2))
            topvorticesy = slope * topvorticesx

        elif config=='dblliner':
            shift = aux['horizontalshift']
            topvorticesx = np.linspace(deltax, linerlength, int(N/4))
            topvorticesy = slope * topvorticesx
            
            topvorticesx = np.concatenate([topvorticesx, shift + topvorticesx])
            topvorticesy = np.concatenate([topvorticesy, topvorticesy])
        
        ###### Set vortex x-y positions  ###################
        x0 = np.concatenate((topvorticesx, topvorticesx))
        y0 = np.concatenate((topvorticesy, -1.*topvorticesy))
    
    elif config=='sine':
        wvn = aux['wavenumbers'][ICnum]
        
        ###### Set vortex x-y positions  ###################
        y0 = np.linspace(0,1,int(N/2))
        y0 = np.concatenate([y0, -y0])
        
        x0 = np.sin((2*np.pi/wvn)*y0)
    
    z0 = np.concatenate((x0, y0, Gamma))
    
    return z0

##########################################################
# Solver Functions
##########################################################

def grab_KLQR_mats(nn, x, B):
    prms = nn.params
    r = prms['ROM_dim']
    Ir = np.identity(r)
    Qpsi = 10e4*Ir
    R = 10e-4
    
    Lambda = nn.K(x, 'apply')
    Bpsi = nn.Psigrad(x, 'apply') * B
    
    return Lambda, Bpsi, Qpsi, R


def dzdt(z, t, incT, B=0, nn=''):
    N = int(len(z)/3)
    
    x = z[0:N]
    y = z[N:2*N]
    Gamma = z[2*N:3*N]

    dxdt = np.zeros(N)
    dydt = np.zeros(N)

    #TODO: Controlled Gammas or uncontrolled?
    for alpha in range(N):
       if incT[alpha]<= t:
          for beta in range(N):
             if alpha != beta:
                lalphabetasq = (x[alpha]-x[beta])**2. + (y[alpha]-y[beta])**2.
                dxdt[alpha] = dxdt[alpha] - 1./(2.*np.pi)*Gamma[beta]*(y[alpha]-y[beta])/lalphabetasq
                dydt[alpha] = dydt[alpha] + 1./(2.*np.pi)*Gamma[beta]*(x[alpha]-x[beta])/lalphabetasq

    #TODO: check this
    # Grab control Bpsi, u
    if nn !='':
        z0 = np.concatenate((x,y,Gamma))
        L, Bpsi, Q, R = grab_KLQR_mats(nn, z0, B)
        u = ctrl.lqr(L, Bpsi, Q, R)
    else:
        B = 0
        u = 0
       
    dzdt = np.concatenate((dxdt,dydt,0*Gamma)) + B*u
    return dzdt


def computevelprofile(z, ylist, t, NumPts):
    N = int(len(z)/3)
    
    x = z[0:N]
    y = z[N:2*N]
    Gamma = z[2*N:3*N]
    
    dxdt = np.zeros(NumPts)
    for alpha in range(N):
      #for kdx in range(NumPts):
        lalphabetasq = (x[alpha])**2. + (y[alpha]-ylist)**2.
        dxdt = dxdt + 1./(2.*PI)*Gamma[alpha]*(y[alpha]-ylist)/lalphabetasq
    return dxdt


def computeMetric(z, ylist, t, NumPts):
    N = int(len(z)/3)
    
    x = z[0:N]
    y = z[N:2*N]
    Gamma = z[2*N:3*N]
    
    dxdt = np.zeros(NumPts)
    for alpha in range(N):
      #for kdx in range(NumPts):
        lalphabetasq = (x[alpha])**2. + (y[alpha]-ylist)**2.
        dxdt = dxdt + 1./(2.*PI)*Gamma[alpha]*(y[alpha]-ylist)/lalphabetasq
    return np.linalg.norm(dxdt)


def solveVGroup(prms, iidx, lists, B=0, nn=''):
    #### Loop over the slope array for generating training data
    ###### Grab initial configuration and Gamma distribution  ###################
    z0 = arrangeV(prms, iidx)

    t = prms['t']
    incT = prms['incT']

    ###### Solve vortex ODEs ###################
    sol = odeint(dzdt, z0, t, args=(incT, B, nn))

    ###### Compute any aditional interesting variables  ###################
    ylist = np.linspace(-0.1,0.1,10)
    NumPts = len(ylist)

    velprofile = computevelprofile(sol[0], ylist, t, NumPts)
    metric = computeMetric(sol[0], ylist, t, NumPts)
       
    lists['vplist'].append(velprofile)
    lists['mlist'].append(metric)
    lists['sollist'].append(sol)
    #paramlist.append([vortexstrength[idx], vortexperiod[jdx], vortexstandoff[kdx], offset[ldx]])

def load_dataset(data_set):
    # Load the appropriate data set
    # FINISH
    dataset = 0
    return dataset
    
def save_dataset(fldr, iidx, dataset):
    print('Saving...')
    train_fn = fldr + "/RMI_data%s.csv" % iidx
    np.savetxt(train_fn, dataset, delimiter=",")
    
    print('Saved.')

##########################################################
# Main solve
##########################################################

#### This file is called by a single worker/node on the cluster
#### It loads its individual slice of the ICs to be generated and solves them
#### Then saves them to a file/number of files and stashes it in the appropriate pool (which may be sorted based on IC params)

def solverange(prms, ICrange, divider):
    # Construct lists
    lists = {'vplist': [], 'mlist': [], 'paramlist': [], 'sollist': []}
    
    for iidx in ICrange: # Batch number
        print('Solving %d of %d' % (iidx+1, len(ICrange)))
        
        solveVGroup(prms, iidx, lists)
        
        if (iidx + 1) % divider == 0:
            sollist = lists['sollist']
            dataset = np.vstack(sollist)
            save_dataset(fldr, iidx+1, dataset)
            lists = {'vplist': [], 'mlist': [], 'paramlist': [], 'sollist': []}
    
    ##### SAVE THE DATA (in the right folder) #####
    sollist = lists['sollist']
    if sollist:
        dataset = np.vstack(sollist)
        save_dataset(fldr, dataset)


# Grab local worker range
loc = str(sys.argv[1])
worknm = int(sys.argv[2])
divider = int(sys.argv[3])
fldr = './data/TVT/' + loc
data_fn = fldr + '/dynsys_params.pkl'
work_fn = fldr + '/worker%s.pkl' % worknm

with open(data_fn, 'rb') as dn:
    prms = pickle.load(dn)
    numICs = prms['numICs']

with open(work_fn, 'rb') as fn:
    work_prms = pickle.load(fn)
    ICrange = work_prms['ICrange']

sollist = solverange(prms, ICrange, divider)