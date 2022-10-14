# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:34:48 2022

@author: gibson48
"""
import os
import pdb
import csv
import time
import json
import copy
import pickle
import pathlib
import helperfns
import numpy as np
import pandas as pd
from random import sample
from scipy.integrate import odeint

import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.losses import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model

from Koopman_config import collapse_params

class NN:
    def __init__(self, parent, prms = dict()):
        self.parent = parent
        self.params = copy.deepcopy(prms)


    def load_model(self, fn=''):
        name = self.params['name']

        if fn == '':
            fn = self.parent.params['model_path']
        
        lastSize = 0 # stupid error catcher
        nw = self.params['num_weights']
        for j in range(nw):
            W1 = np.matrix(np.genfromtxt(fn.replace("model.pkl", "W%s%d.csv" % (name, j + 1)), delimiter=','))
            b1 = np.matrix(np.genfromtxt(fn.replace("model.pkl", "b%s%d.csv" % (name, j + 1)), delimiter=','))
            if j > 0:
                if W1.shape[0] != lastSize:
                    if W1.shape[0] == 1:
                        # need to transpose?
                        if W1.shape[1] == lastSize:
                            W1 = np.transpose(W1)
                        else:
                            print("error: sizes %d and %r" % (lastSize, W1.shape))
                    else:
                        print("error: sizes %d and %r" % (lastSize, W1.shape))
            lastSize = W1.shape[1]

            self.model.layers[j+1].set_weights([np.asarray(W1), np.asarray(b1)[0]])


    def config(self, name, widths, dist_weights, dist_biases, scale, act_type, L1_reg, L2_reg, shifts=[]):
        self.params['name'] = copy.deepcopy(name)
        self.params['widths'] = copy.deepcopy(widths)
        self.params['num_weights'] = len(widths)-1
        self.params['dist_weights'] = copy.deepcopy(dist_weights)
        self.params['dist_biases'] = copy.deepcopy(dist_biases)
        self.params['scale'] = copy.deepcopy(scale)
        self.params['act_type'] = copy.deepcopy(act_type)
        self.params['shifts'] = copy.deepcopy(shifts)
        self.params['num_shifts'] = len(shifts)
        self.params['L1_lam'] = L1_reg
        self.params['L2_lam'] = L2_reg

        self.x = np.ndarray(len(widths)-1,dtype=object)


    def create(self):
        prms = self.params
        name = prms['name']
        widths = prms['widths']
        scale = prms['scale']
        dist_weights = prms['dist_weights']
        dist_biases = prms['dist_biases']
        regs = keras.regularizers.L1L2(l1=prms['L1_lam'], l2=prms['L2_lam'])

        self.x[0] = Input(shape=(widths[0],), dtype=tf.float64)

        mid_widths = widths[1:-1]
        for j, w in enumerate(mid_widths):
            self.x[j+1] = Dense(w, activation=self.params['act_type'], dtype=tf.float64, kernel_regularizer=regs)(self.x[j])
        
        self.y = Dense(widths[-1], dtype=tf.float64, kernel_regularizer=regs)(self.x[-1])
        
        self.model = Model(inputs=self.x[0], outputs=self.y, name=name)
        #(tf.float64, [num_shifts + 1, None, widths[0]])
        return self.x


##### Composite net made of smaller NNs. Omega net is a composite. #####
class LambdaNet:
    def __init__(self, parent, prms=dict()):
        self.params = copy.deepcopy(prms)
        self.parent = parent
        self.count = 0
        self.fixed_output = []
        self.nets = []


    def add_NN(self, prms=dict()):
        # configure the new NN by prms
        self.nets.append(NN(self.parent, prms))
        
        if prms:
            self.nets[-1].config(prms['name'], prms['widths'], dist_weights=prms['dist_weights'], dist_biases=prms['dist_biases'], scale=prms['scale'], L1_reg=prms['L1_lam'], L2_reg=prms['L2_lam'], act_type=prms['act_type'])
        
        self.count += 1


    def form_complex_conjugate_block(self, omegas):
        """Form a 2x2 block for a complex conj. pair of eigenvalues, but for each example, so dimension [None, 2, 2]

        2x2 Block is
        exp(mu * delta_t) * [cos(omega * delta_t), -sin(omega * delta_t)
                             sin(omega * delta_t), cos(omega * delta_t)]

        Arguments:
            omegas -- array of parameters for blocks. first column is freq. (omega) and 2nd is scaling (mu), size [None, 2]
            delta_t -- time step in trajectories from input data

        Returns:
            stack of 2x2 blocks, size [None, 2, 2], where first dimension matches first dimension of omegas

        Side effects:
            None
        """

        dt = self.params['delta_t']

        scale = tf.exp(omegas[:, 1] * dt)
        entry11 = tf.multiply(scale, tf.cos(omegas[:, 0] * dt))
        entry12 = tf.multiply(scale, tf.sin(omegas[:, 0] * dt))
        row1 = tf.stack([entry11, -entry12], axis=1)  # [None, 2]
        row2 = tf.stack([entry12, entry11], axis=1)  # [None, 2]
        blk = tf.stack([row1, row2], axis=2)
        
        return blk # [None, 2, 2] put one row below other


    def Kstep(self, y, omegas):
        dt = self.params['delta_t']

        real = y.shape[1]==1
        if real:
            final = tf.multiply(y[:, np.newaxis], tf.exp(omegas * dt))
            print('Danger, Will Robinson! Danger!')
        else:
            L_stack = self.form_complex_conjugate_block(omegas)
            #ystack = tf.stack([y, y], axis=2)  # [None, 2, 2]            
            #elmtwise_prod = tf.multiply(ystack, L_stack)
            final = tf.linalg.matvec(L_stack, y)#tf.reduce_sum(elmtwise_prod, axis=1)
            #print('Lambdas: ' + str(omegas))
            #print('L_block: ' + str(L_stack))
            #print('z vec: ' + str(y))
            #print('prod: ' + str(elmtwise_prod))
            #print('reduced: ' + str(final))

        return final


    def config(self, name, widths, dist_weights, dist_biases, scale, act_type, L1_reg, L2_reg, shifts=[]):
        self.params['name'] = copy.deepcopy(name)
        self.params['widths'] = copy.deepcopy(widths)
        self.params['num_weights'] = len(widths)-1
        self.params['dist_weights'] = copy.deepcopy(dist_weights)
        self.params['dist_biases'] = copy.deepcopy(dist_biases)
        self.params['scale'] = copy.deepcopy(scale)
        self.params['act_type'] = copy.deepcopy(act_type)
        self.params['shifts'] = copy.deepcopy(shifts)
        self.params['num_shifts'] = len(shifts)
        self.params['L1_lam'] = L1_reg
        self.params['L2_lam'] = L2_reg

        #print(widths)
        prms = self.params
        R, C = prms['ROM_dim']
        self.fixed_output = prms['fixed_omegas']

        temp_prms = dict()
        temp_prms['dist_weights'] = prms['dist_weights_omega']
        temp_prms['dist_biases'] = prms['dist_biases_omega']
        temp_prms['scale'] = prms['scale_omega']
        temp_prms['act_type'] = prms['act_type']
        temp_prms['L1_lam'] = prms['L1_lam']
        temp_prms['L2_lam'] = prms['L2_lam']
        
        for c in np.arange(prms['num_complex_pairs']):
            temp_prms['name'] = 'Lambda_c_%d' % (c + 1)
            temp_prms['widths'] = prms['widths_omega_complex']
            
            self.add_NN(temp_prms)

        for r in np.arange(prms['num_real']):
            temp_prms['name'] = 'Lambda_r_%d' % (r + 1)
            temp_prms['widths'] = prms['widths_omega_real']

            self.add_NN(temp_prms)

        self.x = np.ndarray(len(widths)-1,dtype=object)


    def create(self):
        """Create the auxiliary (omega) network(s), which have ycoords as input and output omegas (parameters for L).
    
        Arguments:
            params -- dictionary of parameters for experiment
            ycoords -- array of shape [None, k] of y-coordinates, where L will be k x k
    
        Returns:
            omegas -- list, output of omega (auxiliary) network(s) applied to input ycoords
            weights -- dictionary of weights
            biases -- dictionary of biases
    
        Side effects:
            Adds 'num_omega_weights' key to params dict
        """
        prms = self.params
        widths = prms['widths']
        temp_prms = dict()
        temp_prms['dist_weights'] = prms['dist_weights_omega']
        temp_prms['dist_biases'] = prms['dist_biases_omega']
        temp_prms['scale'] = prms['scale_omega']
        temp_prms['act_type'] = prms['act_type']
        temp_prms['L1_lam'] = prms['L1_lam']
        temp_prms['L2_lam'] = prms['L2_lam']

        temp_out = []
        K_mdls = []
        self.slc = []
        self.nrm = []
        self.K_step = []
        self.Lcmp = []
        self.dims = prms['ROM_dim']
        R, C = self.params['ROM_dim']
        self.fixed_output = prms['fixed_omegas']

        self.x[0] = Input(shape=(widths[0],), dtype=tf.float64)

        for c in np.arange(prms['num_complex_pairs']):
            cidx1 = 2*c
            cidx2 = 2*(c+1)
            
            self.nets[c].create()

            self.slc.append(Lambda(lambda x,k=c: x[:,2*k:2*(k+1)], dtype=tf.float64, name='Slice_%s-%s' % (2*c, 2*(c+1)) )(self.x[0]))
            self.nrm.append(Lambda(lambda x: tf.norm(x, ord='euclidean', axis=1, keepdims=True), dtype=tf.float64, name='Take_Norm_%s-%s' % (cidx1, cidx2))(self.slc[-1]))
            temp_out.append(self.nets[c].model(self.nrm[-1]))
            
            self.K_step.append(Lambda(lambda x: self.Kstep(x[0], x[1]), dtype=tf.float64)([self.slc[-1], temp_out[-1]]))
            K_mdls.append(self.K_step[-1])

        for r in np.arange(prms['num_real']):
            ridx = 2*C + r

            self.slc.append(Lambda(lambda x: x[:,ridx], dtype=tf.float64, name='Slice_%s' % (ridx))(self.x[0]))
            
            temp_out.append(self.nets[c+r].model(self.slc[-1]))
            
            self.K_step.append(Lambda(lambda x: self.Kstep(x[0], x[1]), dtype=tf.float64)([self.slc[-1], temp_out[-1]]))
            K_mdls.append(self.K_step[-1])

        #### Add in the fixed_omega Lambda layers! tf.constant!
        cmp = []
        Tidx = R + 2*C
        #print(prms['fixed_omegas'])
        for fidx, fixed_L in enumerate(prms['fixed_omegas']):
            fidx1 = Tidx + 2*fidx
            fidx2 = fidx1 + 2
            self.slc.append(Lambda(lambda x,k=fidx1: x[:,k:(k+2)], dtype=tf.float64, name='Slice_%s-%s' % (fidx1, fidx2))(self.x[0]))

            cmp.append(tf.convert_to_tensor(np.array([[np.imag(fixed_L), np.real(fixed_L)]]), dtype=tf.float64))
            self.Lcmp.append(Lambda(lambda x,k=fidx: tf.repeat(cmp[k], repeats=tf.shape(x)[0], axis=0), name='Fixed_Lambda_%s-%s' % (fidx1, fidx2))(self.slc[-1]))
            temp_out.append(self.Lcmp[-1])
            
            self.K_step.append(Lambda(lambda x: self.Kstep(x[0], x[1]), dtype=tf.float64)([self.slc[-1], temp_out[-1]]))
            K_mdls.append(self.K_step[-1])

        self.y = np.ndarray(len(temp_out)+1,dtype=object)
        self.Lmodel = np.ndarray(len(temp_out)+1,dtype=object)
        
        self.y[0] = Concatenate(axis=1, dtype=tf.float64)(temp_out)
        self.Lmodel[0] = Model(inputs=self.x[0], outputs=self.y[0], name='Lambda_Network')
        #plot_model(self.Lmodel[0], to_file='Lmodel_plot_0.png', show_shapes=True, show_layer_names=True, expand_nested=True)
        
        for tidx, tmp in enumerate(temp_out):
            self.y[tidx+1] = tmp
            self.Lmodel[tidx+1] = Model(inputs=self.x[0], outputs=self.y[tidx+1], name='Lambda_Network')
            plot_model(self.Lmodel[tidx+1], to_file='Lmodel_plot_%d.png' % (tidx+1), show_shapes=True, show_layer_names=True, expand_nested=True)
        
        self.Ky = Concatenate(axis=1, dtype=tf.float64)(K_mdls)
        self.Kmodel = Model(inputs=self.x[0], outputs=self.Ky, name="Koopman_Dynamics")
        
        
        #plot_model(self.Kmodel, to_file='Kmodel_plot.png', show_shapes=True, show_layer_names=True, expand_nested=True)



#class Kstep:
#    def __init__(self, units=32, input_dim=32):
#        super(Kstep, self).__init__()
#        w_init = tf.random_normal_initializer()
#        self.w = tf.Variable(
#            initial_value=w_init(shape=(input_dim, units), dtype="float64"),
#            trainable=True,
#        )
#        b_init = tf.zeros_initializer()
#        self.b = tf.Variable(
#            initial_value=b_init(shape=(units,), dtype="float64"), trainable=True
#        )

#    def call(self, inputs):
#        return tf.matmul(inputs, self.w) + self.b


##### Main Koopman Neural Net #####
class KoopmanNet:
    # instance attribute
    def __init__(self, params):
        self.params = copy.deepcopy(params)
        self.config()


    def config(self):
        prms = self.params

        # grab its location on the cluster and configure node environment
        self.worker = worker(self, prms)

        # 2 psi nets and 1 lambda net
        self.encoder = NN(self, prms)
        self.decoder = NN(self, prms)
        self.lambdanet = LambdaNet(self, prms)
        
        if 'seed' in prms.keys():
            tf.random.set_seed(prms['seed'])
            np.random.seed(prms['seed'])

        # split up widths, shifts and other parameters into the encoder, decoder and omega networks
        depth = int((prms['d'] - 4) / 2)
        encoder_widths = prms['widths'][0:depth + 2]  # n ... k
        decoder_widths = prms['widths'][depth + 2:]  # k ... n
        lambda_widths = [encoder_widths[-1], prms['hidden_widths_omega'][0], decoder_widths[0]]
        
        #max_shifts_to_stack = helperfns.num_shifts_in_stack(prms)
        self.encoder.config('Psi', encoder_widths, dist_weights=prms['dist_weights'][0:depth + 1], dist_biases=prms['dist_biases'][0:depth + 1], scale=prms['scale'], act_type=prms['act_type'], L1_reg=prms['L1_lam'], L2_reg=prms['L2_lam'], shifts=prms['shifts_middle'])
        self.decoder.config('Psi_Inv', decoder_widths, dist_weights=prms['dist_weights'][depth + 2:], dist_biases=prms['dist_biases'][depth + 2:], scale=prms['scale'], act_type=prms['act_type'], L1_reg=prms['L1_lam'], L2_reg=prms['L2_lam'], shifts=prms['shifts_middle'])
        self.lambdanet.config('Lambda', lambda_widths, dist_weights=prms['dist_weights_omega'], dist_biases=prms['dist_biases_omega'], scale=prms['scale_omega'], act_type=prms['act_type'], L1_reg=prms['L1_lam'], L2_reg=prms['L2_lam'], shifts=prms['shifts'])


    def define_loss(self):
        """Define the (unregularized) loss functions for the training.

        Arguments:
            x -- placeholder for input
            xstar -- list of outputs of network for each shift (each prediction step)
            ytilde -- list of output of encoder for each shift (encoding each step in x)
            weights -- dictionary of weights for all networks
            biases -- dictionary of biases for all networks
            params -- dictionary of parameters for experiment

        Returns:
            loss1 -- autoencoder loss function
            loss2 -- dynamics/prediction loss function
            loss3 -- linearity loss function
            loss_Linf -- inf norm on autoencoder loss and one-step prediction loss
            loss -- sum of above four losses

        Side effects:
            None
        """
        # Minimize the mean squared errors.
        # subtraction and squaring element-wise, then average over both dimensions
        # n columns
        # average of each row (across columns), then average the rows
        prms = self.params
        denominator_nonzero = 10 ** (-5)
        
        x = self.encoder.x[0]
        xstar = self.model(x) # Encode, advance, and decode an entire stack x
        ytilde = self.encoder.model(x) # Encode. That's it. An entire stack x

        # autoencoder loss
        if prms['relative_loss']:
            loss1_denominator = tf.norm(x[0], ord='euclidean', keepdims=True) + denominator_nonzero
        else:
            loss1_denominator = tf.cast(1.0, tf.double)

        mean_squared_error = mse(x[:,0], ytilde[:,0])#tf.reduce_mean(tf.reduce_mean(tf.square(xstar[0] - tf.squeeze(x[0, :, :])), 1))
        loss1 = prms['recon_lam'] * tf.truediv(mean_squared_error, loss1_denominator)
        
        

        # gets dynamics/prediction
        loss2 = tf.zeros([1, ], dtype=tf.float64)
        if prms['num_shifts'] > 0:
            for j in np.arange(prms['num_shifts']):
                # xk+1, xk+2, xk+3
                shift = prms['shifts'][j]
                if prms['relative_loss']:
                    loss2_denominator = tf.norm(x[shift], ord='euclidean', keepdims=True) + denominator_nonzero
                else:
                    loss2_denominator = tf.cast(1.0, tf.double)

                mean_squared_error = mse(x[shift], xstar[j+1])#tf.reduce_mean(tf.reduce_mean(tf.square(xstar[j + 1] - tf.squeeze(x[shift, :, :])), 1))
                loss2 = loss2 + prms['recon_lam'] * tf.truediv(mean_squared_error, loss2_denominator)

            loss2 = loss2 / prms['num_shifts']

        # K linear
        loss3 = tf.zeros([1, ], dtype=tf.float64)
        count_shifts_middle = 0
        if prms['num_shifts_middle'] > 0:
            # generalization of: next_step = tf.matmul(g_list[0], L_pow)
            omegas = self.lambdanet.model(ytilde)
            next_step = self.K_step(ytilde, omegas)
            
            # multiply g_list[0] by L (j+1) times
            for j in np.arange(max(prms['shifts_middle'])):
                if (j + 1) in prms['shifts_middle']:
                    if prms['relative_loss']:
                        loss3_denominator = tf.norm(ytilde[count_shifts_middle + 1,:], ord='euclidean', axis=1, keepdims=True) + denominator_nonzero#tf.reduce_mean(tf.reduce_mean(tf.square(tf.squeeze(ytilde[count_shifts_middle + 1])), 1)) + denominator_nonzero
                    else:
                        loss3_denominator = tf.cast(1.0, tf.double)

                    mean_squared_error = mse(next_step, ytilde[count_shifts_middle + 1,:])
                    loss3 = loss3 + prms['mid_shift_lam'] * tf.truediv(mean_squared_error, loss3_denominator)
                    count_shifts_middle += 1

                omegas = self.lambdanet.model(next_step)
                next_step = self.K_step(next_step, omegas)

            loss3 = loss3 / prms['num_shifts_middle']

        # inf norm on autoencoder error and one prediction step
        if prms['relative_loss']:
            Linf1_den = tf.norm(tf.norm(tf.squeeze(x[0, :, :]), axis=1, ord=np.inf), ord=np.inf) + denominator_nonzero
            Linf2_den = tf.norm(tf.norm(tf.squeeze(x[1, :, :]), axis=1, ord=np.inf), ord=np.inf) + denominator_nonzero
        else:
            Linf1_den = tf.cast(1.0, tf.double)
            Linf2_den = tf.cast(1.0, tf.double)

        Linf1_penalty = tf.truediv(tf.norm(tf.norm(xstar[0] - tf.squeeze(x[0, :, :]), axis=1, ord=np.inf), ord=np.inf), Linf1_den)
        Linf2_penalty = tf.truediv(tf.norm(tf.norm(xstar[1] - tf.squeeze(x[1, :, :]), axis=1, ord=np.inf), ord=np.inf), Linf2_den)
        loss_Linf = prms['Linf_lam'] * (Linf1_penalty + Linf2_penalty)

        loss = loss1 + loss2 + loss3 + loss_Linf

        return loss1, loss2, loss3, loss_Linf, loss


    def clear(self):
        self.encoder = NN(self, self.params)
        self.decoder = NN(self, self.params)
        self.lambdanet = LambdaNet(self, self.params)


    def save_model(self, sess, csv_path, train_val_error):
        """Save error files, weights, biases, and parameters.

        Arguments:
            sess -- TensorFlow session
            csv_path -- string for path to save error file as csv
            train_val_error -- table of training and validation errors
            params -- dictionary of parameters for experiment
            weights -- dictionary of weights for all networks
            biases -- dictionary of biases for all networks

        Returns:
            None (but side effect of saving files and updating params dict.)

        Side effects:
            Save train_val_error, each weight W, each bias b, and params dict to file.
            Update params dict: minTrain, minTest, minRegTrain, minRegTest
        """
        np.savetxt(csv_path, train_val_error, delimiter=',')
        params = self.params
        weights, biases = self.grab_net()

        for key, value in weights.items():
            np.savetxt(csv_path.replace('error', key), np.asarray(sess.run(value)), delimiter=',')
        for key, value in biases.items():
            np.savetxt(csv_path.replace('error', key), np.asarray(sess.run(value)), delimiter=',')

        params['minTrain'] = np.min(train_val_error[:, 0])
        params['minTest'] = np.min(train_val_error[:, 1])
        params['minRegTrain'] = np.min(train_val_error[:, 2])
        params['minRegTest'] = np.min(train_val_error[:, 3])
        print("min train: %.12f, min val: %.12f, min reg. train: %.12f, min reg. val: %.12f" % (
            params['minTrain'], params['minTest'], params['minRegTrain'], params['minRegTest']))

        self.save_params()


    def save_params(self):
        """Save parameter dictionary to file.

        Arguments:
            params -- dictionary of parameters for experiment

        Returns:
            None

        Side effects:
            Saves params dict to pkl file
        """
        params = self.params
        with open(params['model_path'].replace('ckpt', 'pkl'), 'wb') as f:
            pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)


    def load_params(self, fn, verbose='off'):
        pkl_fn = fn

        with open(pkl_fn, 'rb') as fn:
            params = pickle.load(fn)

        if verbose=='on':
            return params

        self.params = collapse_params(params)
        self.config()


    def load_model(self, fn=''):        
        # grab file locations
        if fn == '':
            fn = self.params['model_path']
        else:
            self.load_params(fn)
            self.params['model_path'] = fn

        # load weights & biases and distribute them to subnetworks
        self.encoder.load_model()
        self.decoder.load_model()
        for j in np.arange(self.lambdanet.count):
            self.lambdanet.nets[j].load_model()


    def save(self, fldr):
#        './output/80-80-170'
        self.encoder.model.save(fldr + '/encoder/')
        self.decoder.model.save(fldr + '/decoder/')
        self.lambdanet.Lmodel[0].save(fldr + '/Lmodel/')
        self.lambdanet.Kmodel.save_weights(fldr + '/Kmodel/')
        
        
    def load(self, fldr):
        self.encoder.model = keras.models.load_model(fldr + '/encoder/')
        self.decoder.model = keras.models.load_model(fldr + '/decoder/')
        self.lambdanet.Lmodel = keras.models.load_model(fldr + '/Lmodel/')
        self.lambdanet.Kmodel = tf.saved_model.load(fldr + '/Kmodel/')


    def load_data(self, data_fn):
        
        return

    def predict(self, x0, time_steps):
        """Use the Koopman net to predict forward in time from an initial condition
        
        Arguments:
            x0 - initial condition, a single data-point x, must be a list of float64s of length self.params['widths'][0]
            time_steps - number of time-steps to predict into the future, integer

        Returns:
            Arrays of length time_steps.
            x - Dynamical states
            z - Koopman states
            L - Lambdas at each time-step
        """

        x = []
        z = []
        L = []
        
        x.append(x0)
        z.append(self.encoder.model(x[-1]).numpy())
        L.append(self.lambdanet.Lmodel[0](z[-1]).numpy())

        # Step forward in time through the Koopman states and then decode back to dynamics
        for t in np.arange(time_steps):
            z.append(self.lambdanet.Kmodel(z[-1]).numpy())
            L.append(self.lambdanet.Lmodel[0](z[-1]).numpy())
            x.append(self.decoder.model(z[-1]).numpy())
        
        x = np.vstack(x)
        z = np.vstack(z)
        L = np.vstack(L)
        
        return x, z, L

    def create(self):
        """Create a Koopman network that encodes, advances in time, and decodes.

        Arguments:
            none -- internal settings

        Returns:
            x -- placeholder for input. An entire stack with length num_shifts_middle + 1.
            ytilde -- list, output of encoder applied to each timestep in input x, length num_shifts_middle + 1
            ystar -- list, output of encoder applied to first timestep in input x, then advanced m times by K, where m is length num_shifts + 1
            omegas -- list, output of lambdanet during the time prediction in Koopman states, i.e. what goes into K, K^2, K^3, ..., K^m
            xtilde -- list, output of decoder applied to ytilde. Reconstruction of x.
            xstar -- list, output of decoder applied to ystar. Predicted dynamics.
            
            x is a dynamic timeseries, ytilde is the corresponding Koopman states, xtilde is the reconstruction (perfect \psi^{-1}(\psi(x)) means x = xtilde).
            ystar is the predicted evolution of Koopman states starting from initial condition \psi(x[0]), and xstar is its reconstruction \psi^{-1}(ystar), i.e. the predicted dynamic timeseries.
            Lot of information floating around here! Broadly:
                tilde is reconstruction, star is prediction. x is dynamic states, y is Koopman states.

        Side effects:
            Adds more entries to params dict: num_encoder_weights, num_omega_weights, num_decoder_weights

        Raises ValueError if len(y) is not len(params['shifts']) + 1
        """
        # create the encoder, decoder and omega NNs. MUST BE CONFIGURED FIRST USING self.config()
        self.encoder.create()
        self.decoder.create()
        self.lambdanet.create()
        
        prms = self.params
        T = prms['len_time']
        widths = prms['widths']
        mid_shifts = prms['shifts']
        blk_len = len(mid_shifts)+1
        
        x = Input(shape=(T, widths[0]), dtype=tf.float64)
        x0 = Input(shape=(widths[0],), dtype=tf.float64)
        x_blk = x[:,0:blk_len,:]

        ytilde_temp = []
        xtilde_temp = []
        for tidx in np.arange(T):
            ytilde_temp.append(self.encoder.model(tf.squeeze(x[:,tidx,:])))
            xtilde_temp.append(self.decoder.model(ytilde_temp[-1]))

        ytilde = tf.stack(ytilde_temp, axis=1, name='Encoded_Timeseries')
        xtilde = tf.stack(xtilde_temp, axis=1, name='Reconstructed_Timeseries')
        ytilde_blk = ytilde[:,0:blk_len,:]
        
        ystar_temp = []
        xstar_temp = []
        ystar_temp.append(tf.squeeze(ytilde[:,0,:]))
        xstar_temp.append(tf.squeeze(x[:,0,:]))
        for shidx, sh in enumerate(mid_shifts):
            ystar_temp.append(self.lambdanet.Kmodel(ystar_temp[-1]))
            xstar_temp.append(self.decoder.model(ystar_temp[-1]))

        ystar = tf.stack(ystar_temp, axis=1, name='Predicted_Koopman_States')
        xstar = tf.stack(xstar_temp, axis=1, name='Predicted_Dynamic_States')
        
        #loss1, loss2, loss3, loss_Linf, loss = self.define_loss()
        #loss_L1, loss_L2, regularized_loss, regularized_loss1 = self.define_regularization(loss, loss1)
        
        self.encoder_decoder = Model(inputs=x, outputs=xtilde)
        self.single = Model(inputs=x0, outputs=self.decoder.model(self.lambdanet.Kmodel(self.encoder.model(x0))))
        self.model = Model(inputs=x, outputs=xstar)
        
        reconstruction_loss = tf.reduce_mean(mse(x, xtilde))
        state_prediction_loss = tf.reduce_mean(mse(x_blk, xstar))
        linear_dynamics_loss = tf.reduce_mean(mse(ytilde_blk, ystar))
        loss = reconstruction_loss + state_prediction_loss + linear_dynamics_loss
        
        train_var = self.model.trainable_variables
        #l1_regularizer = tf.add_n([tf.nn.l1_loss(v) for v in train_var if 'b' not in v.name])
        loss_L1 = prms['L1_lam']
        l2_regularizer = tf.add_n([tf.nn.l2_loss(v) for v in train_var if 'b' not in v.name])
        loss_L2 = prms['L2_lam'] * l2_regularizer

        reg_loss = loss + loss_L1 + loss_L2

        self.model.add_loss(reg_loss)
        
        self.model.add_metric(reconstruction_loss, name='reconstruction_loss')
        self.model.add_metric(state_prediction_loss, name='state_prediction_loss')
        self.model.add_metric(linear_dynamics_loss, name='linear_dynamics_loss')
        self.model.add_metric(reg_loss, name='regularized_loss')
        self.model.compile(optimizer='adam')
        
        return self.model


    def train(self, epochs=2500, batch_size=128, work_dict={}):
        """Run a random experiment for particular params and data.

        Arguments:
            data_val -- array containing validation dataset
            params -- dictionary of parameters for experiment

        Returns:
            None

        Side effects:
            Changes params dict
            Saves files
            Builds TensorFlow graph (reset in main_exp)
        """
        if work_dict: self.worker.config(work_dict)
        self.worker.config_ops()
        
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        print(strategy.num_replicas_in_sync)

        with strategy.scope():
            model = self.create()

        local_path = self.worker.params['paths']['local']
        tb_path = self.worker.params['paths']['tensorboard']
        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath=local_path, save_freq=100),
            keras.callbacks.TensorBoard(tb_path)
        ]
        
        data_train, data_val, data_test = self.grab_data()

        # Distributable training steps
        print('Beginning training...')
        history = model.fit(data_train, batch_size=batch_size, callbacks=callbacks, epochs=epochs, validation_data=(data_val), verbose=1)
        
        return history


    def grab_data(self):
        data_set = []
        prms = self.params
        csv_path = prms['model_path'].replace('model', 'error')
        csv_path = csv_path.replace('ckpt', 'csv')
        print('Model: ' + csv_path)
        
        print('Loading data...')
        data_path = './data/TVT/' + prms['train_data']
            
        L = prms['numICs']
        B = L/prms['batch_size']
        jumpsize = 50
        nums = np.arange(L)[(jumpsize-1):L:jumpsize]
        num_train = int(np.floor(.7*B))
        num_val = int(np.floor(.2*B))
        num_test = int(np.floor(.2*B))

        for file_num in nums:
            data_set.append(np.loadtxt(('./data/TVT/%s/%s_data%d.csv' % (prms['train_data'], prms['data_name'], file_num)), delimiter=',', dtype=np.float64))

        data_set = np.vstack(data_set)
        data_set = np.reshape(data_set, (int(data_set.shape[0]/101), 101, 108))
            
        dataset = tf.data.Dataset.from_tensor_slices(data_set).batch(prms['batch_size']).shuffle(buffer_size=prms['batch_size'], seed=prms['seed'])
        data_train = dataset.take(num_train)
        data_val = dataset.skip(num_train).take(num_val)
        data_test = dataset.skip(num_train+num_val).take(num_test)
            
        return data_train, data_val, data_test


class worker:
    def __init__(self, parent, prms=dict()):
        self.parent = parent
        
        if prms:
            self.create(prms)

    #### Generate worker settings and configure based on NN parameters
    def create(self, prms):
        work_dict = {'worker': {'list': ["localhost:12345", "localhost:23456"], 'task': {'type': 'worker', 'index': 0}}}
        
        work_dict['paths']['']
        self.config(work_dict)


    def config(self, prms):
        if 'worker' in prms.keys():
            self.params = prms
            
            if self.params['worker']['task']['index'] == 0:
                self.params['paths']['local'] = 'path/to/cloud/location/ckpt'
                self.params['paths']['tensorboard'] = 'path/to/cloud/location/tb/'
            else:
                self.params['paths']['local']  = 'local/path/ckpt'
                self.params['paths']['tensorboard'] = 'local/path/tb/'

            self.environment = json.dumps({'cluster': {'worker': self.params['worker']['list']}, 'task': {'type': self.params['worker']['task']['type'], 'index': self.params['worker']['task']['index']}})
        #'cluster': {'worker': ["localhost:12345", "localhost:23456"]}, 'task': {'type': 'worker', 'index': 0}


    def config_ops(self):
        os.environ['TF_CONFIG'] = self.environment
        return self.environment
        

class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}