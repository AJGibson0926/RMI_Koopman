# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 16:03:41 2022

@author: gibson48
"""

import os
import pickle
import numpy as np
##########################################################
# Main config
##########################################################

def extract_prms(args):
    idx = args[0]
    if '-' in idx:
        widths = idx.split('-')
        widths = list(map(int, widths))
    else:
        widths = grab_hypers(int(idx))

    if len(args) > 2:
        lr = float(args[1])
        data_set = 'data/TVT/' + args[2] + '/dynsys_params.pkl'
        keigfn = args[3].split(',')
        keigfn[0] = int(keigfn[0])
        keigfn[1] = int(keigfn[1])
        keigfn[2:] = [list(map(complex, keigfn[2:]))]
        tm = float(args[4])
    else: # Defaults
        lr = 0.001
        data_set = str(0) # Use first dataset
        keigfn = [0,2]
        tm = 1
    
    return data_set, widths, lr, keigfn, tm

    
def grab_hypers(idx):
    hyp_fn = 'RMI_hyperparams.csv'
    hyperparams = np.genfromtxt(hyp_fn, delimiter=',', dtype=int)
    
    # change this to a single-line-only read instead of loading the entire thing into memory each time. Not a big savings, but still.
    widths = hyperparams[idx,:]
    return widths

##### Generate circulations Gamma #####
def makeGamma(option, numvort, vortexstrength, noise='off'):
    n = int(numvort/2)
    if option == 0:  # constant vorticity
        Gamma = np.concatenate((np.ones(n), -1.*np.ones(n))) * vortexstrength
    elif option == 1: # Decay function exponential
        decayExp = np.exp(-np.linspace(0,1,n))
        Gamma = np.concatenate((decayExp,-decayExp)) * vortexstrength
    elif option == 2: # sine wave
        # SOMETHING
        #wvn = prms['aux']['wavenumbers'][ICnum]
        #Gamma = np.cos()
        return
    else:
        decayExp = np.exp(-np.linspace(0,1,n))
        Gamma = np.concatenate((decayExp,-decayExp)) * vortexstrength
    
    if noise=='on':
        mu, sigma = 1, 0.1
        ns = np.random.normal(mu, sigma, size=(1,len(Gamma)))
        Gamma = Gamma*ns
    
    return Gamma

##### DYNAMICAL SYSTEM CONFIG #####
def grab_aux(option, prms, noise='off'):
    aux = {}

    # Physical parameters
    shockSpeed = 5.6

    if option=='liner' or option=='dblliner':
        deltax = 0.1
        linerlength = 1.8
        
        aux['deltax'] = deltax
        aux['linearlength'] = linerlength
        
        numvort = 2*int((linerlength - deltax)/deltax + 1)

        vortexCirculationPerLength = 1.
        vortexstrength = [vortexCirculationPerLength * deltax]
        
        Gamma = makeGamma(1, numvort, vortexstrength, noise=noise)
        if option=='dblliner':
            numvort = 2*numvort
            Gamma = np.concatenate([Gamma/2, -Gamma/2])
            aux['horizontalshift'] = -1

        ###### Set an incubation time in case we want to delay the motion fo a vortex point  ###################
        timeMultiplier = linerlength/shockSpeed
        incTupper =  np.linspace(0,0,numvort)*timeMultiplier
        incT = np.concatenate((incTupper,incTupper))
        
        mintheta = 12
        maxtheta = 75
        dtheta = 0.01
        Ntheta = int((maxtheta - mintheta)/dtheta)
        
        aux['mintheta'] = mintheta
        aux['maxtheta'] = maxtheta
        aux['Ntheta'] = Ntheta
    
        slopes = np.linspace(mintheta, maxtheta, Ntheta) * (np.pi/180);
        numICs = len(slopes);
        
        aux['slopes'] = slopes
        
    elif option=='sine':
        deltay = 0.1
        wavelength = 1.8
        
        aux['deltay'] = deltay
        aux['wavelength'] = wavelength
        
        numvort = 2*int((wavelength - deltay)/deltay + 1)
        
        vortexCirculationPerLength = 1.
        vortexstrength = [vortexCirculationPerLength * deltay]
        Gamma = makeGamma(1, numvort, vortexstrength, 1)
        
        ###### Set an incubation time in case we want to delay the motion fo a vortex point  ###################
        timeMultiplier = 1/shockSpeed
        incTupper =  np.linspace(0,0,numvort)*timeMultiplier
        incT = np.concatenate((incTupper,incTupper))

        minwvn = 1
        maxwvn = 10
        dw = 0.001
        Nwvn = int((maxwvn - minwvn)/dw)
        wvns = np.linspace(minwvn, maxwvn, Nwvn)
        
        aux['minwavenumber'] = minwvn
        aux['maxwavenumber'] = maxwvn
        aux['Nwavenumbers'] = Nwvn
        
        numICs = len(wvns);
        
        aux['wavenumbers'] = wvns

    elif option=='polygon': # circularly arrange the vortices, i.e. points of a polygon
        return
    
    prms['numICs'] = numICs # per training file
    
    prms['numvort'] = numvort
    prms['Gamma'] = Gamma
    prms['ShockSpeed'] = shockSpeed
    prms['incT'] = incT
    
    prms['aux'] = aux
    
    return

def config_dynsys(option, noise='off'):
    dynsys_params = {}

    dynsys_params['config'] = option
    dynsys_params['data_name'] = 'RMI'

    # Computation parameters
    numTimeSteps = 101
    runTime = 5   
    t = np.linspace(0, runTime, numTimeSteps)
    
    dynsys_params['len_time'] = numTimeSteps
    dynsys_params['delta_t'] = t[1] - t[0]
    dynsys_params['t'] = t
    dynsys_params['data_train_len'] = 3

    grab_aux(option, dynsys_params, noise=noise)
    
    dynsys_params['data_id'] = dynsys_params['data_name'] + '-' + dynsys_params['config'] + '-' + str(numTimeSteps) + '-' + str(dynsys_params['numvort'])
    dynsys_params['dim'] = 3*dynsys_params['numvort']
    
    return dynsys_params

##### NEURAL NET CONFIG #####
def config_NN(dynsys_params, widths, keigfn, lr, tm):
    NN_params = {}

    # settings related to saving results
    NN_params['id'] = '-'.join(str(e) for e in widths)
    NN_params['folder_name'] = 'output/' + NN_params['id']
    NN_params['train_data'] = dynsys_params['data_id']
    
    # settings related to network architecture
    if any(isinstance(i, list) for i in keigfn):
        NN_params['fixed_omegas'] = keigfn[-1]
    else:
        NN_params['fixed_omegas'] = ''
    
    if type(widths)==np.ndarray:
        core_widths = [widths[1]]*(widths[0])
        w1 = widths[3]
    else:
        core_widths = widths[:-1]
        w1 = widths[-1:][0]

    NN_params['ROM_dim'] = [keigfn[0],keigfn[1]]
    NN_params['num_real'] = keigfn[0]
    NN_params['num_complex_pairs'] = keigfn[1]
    NN_params['num_evals'] = NN_params['num_real'] + 2*(NN_params['num_complex_pairs'] + len(NN_params['fixed_omegas']))
    k = NN_params['num_evals']  # dimension of y-coordinates: size of K-ROM, i.e. number of eigenfunctions to use.
    n = 3*dynsys_params['numvort'] # (x_alpha, y_alpha, Gamma_alpha)
    
    NN_params['widths'] = np.concatenate(([n], core_widths, [k, k], list(reversed(core_widths)), [n]))
    NN_params['hidden_widths_omega'] = [w1,]
    
    # defaults related to initialization of parameters
    NN_params['dist_weights'] = 'dl'
    NN_params['dist_weights_omega'] = 'dl'
    
    # settings related to loss function
    NN_params['num_shifts'] = 30
    NN_params['num_shifts_middle'] = dynsys_params['len_time'] - 1

    NN_params['recon_lam'] = .001
    NN_params['Linf_lam'] = 10 ** (-9)
    NN_params['L1_lam'] = 0.0
    NN_params['L2_lam'] = 10 ** (-14)
    NN_params['auto_first'] = 1
    
    # settings related to training
    NN_params['learning_rate'] = lr
    
    NN_params['num_passes_per_file'] = 15 * 6 * 50
    NN_params['num_steps_per_batch'] = 2
    
    NN_params['batch_size'] = 128
    
    numICs = dynsys_params['numICs']
    max_shifts = max(NN_params['num_shifts'], NN_params['num_shifts_middle'])
    num_examples = numICs * (dynsys_params['len_time'] - max_shifts)
    steps_to_see_all = num_examples / NN_params['batch_size']
    
    NN_params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * NN_params['num_steps_per_batch']
    
    # settings related to timing
    NN_params['max_time'] = tm * 60 * 60  # 4 hours for each run
    NN_params['min_5min'] = 1
    NN_params['min_20min'] = 1
    NN_params['min_40min'] = 1
    NN_params['min_1hr'] = 1
    NN_params['min_2hr'] = 1
    NN_params['min_3hr'] = 1
    NN_params['min_4hr'] = 1
    NN_params['min_halfway'] = 1
    
    params = {**NN_params, **dynsys_params}
    print(params)
    
    return params

def load_params(sources, option): # Make option optional
    params = []
    for src in sources:
        with open(src, 'rb') as f:
            prm = pickle.load(f, encoding='latin1')
            if option=='dynsys':
                params.append(prm['dynsys_params'])
            elif option=='NN':
                params.append(prm['NN_params'])
            elif option=='both': #both, collapse and grab all of it
                params.append(collapse_params(prm))
            else:
                params.append(prm)

    return params

def collapse_params(prms):
    if 'dynsys_params' in prms.keys():
        params = {**prms['dynsys_params'], **prms['NN_params']}
    else:
        params = prms
    
    return params

def save_params(fldr, prms):
    prm_fn = fldr + "/dynsys_params.pkl"
    os.makedirs(os.path.dirname(prm_fn), exist_ok=True)
    with open(prm_fn, 'wb') as dn:
        pickle.dump(prms, dn)
    
def config(option, widths, keigfn, lr, tm):
    
    dynsys_params = config_dynsys(option)
    NN_params = config_NN(dynsys_params, widths, keigfn, lr, tm)
    
    params = {'dynsys_params': dynsys_params, 'NN_params': NN_params}
    
    return params