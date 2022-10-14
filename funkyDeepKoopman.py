# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 12:14:05 2022

@author: gibson48
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Hyperparameters
RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Network and training hyperparameters
NUM_TIME_STEPS = 30 # number of time steps > 1 to include in linearity/prediction loss
BATCH_SIZE = 256
EPOCHS = 30
HIDDEN_DIM = 50
LATENT_DIM = 2

# loss hyperparameters
RECON_LOSS_WEIGHT = 1e-1

# Load dataset
# The datset  was generated from helper scripts in the 
df = pd.read_csv('./data/old/DiscreteSpectrumExample_train1_x.csv')
x = df.values

# Split single long time series into training and test
x_train, x_test = train_test_split(x, test_size=0.2)

# Time-shift data
x_train = tf.keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    None,
    sequence_length=NUM_TIME_STEPS,
    batch_size=BATCH_SIZE,
)
x_test = tf.keras.preprocessing.timeseries_dataset_from_array(
    x_test,
    None,
    sequence_length=NUM_TIME_STEPS,
    batch_size=BATCH_SIZE
)

# Convert to numpy
x_train = np.concatenate(list(x_train.as_numpy_iterator()))
x_test =  np.concatenate(list(x_test.as_numpy_iterator()))
print('x_train.shape', x_train.shape)
print('x_test.shape', x_test.shape)

# batch_input_shape = next(x_train.as_numpy_iterator()).shape
# input_shape = batch_input_shape[1:]
input_shape = x_train.shape[1:]
assert len(input_shape) == 2, 'Assuming that state dimension is flat'
sequence_length, state_dims = input_shape

# print('batch input shape:', batch_input_shape)
print('sequence length:', sequence_length)
print('state dimensions:', state_dims)

# Autoencoder and linear-dynamics network definition

# Encodes input to low-dimensional code, flattening as necessary
encoder = tf.keras.Sequential(
    [
     tf.keras.layers.Dense(HIDDEN_DIM, activation=tf.nn.relu, name='encoder_hidden'),
     tf.keras.layers.Dense(LATENT_DIM, activation=tf.nn.sigmoid, name='code')
    ],
    name='encoder'
)

# Decodes from low-dimensional code to output, handling any reshaping as necessary
decoder = tf.keras.Sequential(
    [
      tf.keras.layers.Dense(HIDDEN_DIM, activation=tf.nn.relu, name='decoder_hidden'),
      tf.keras.layers.Dense(state_dims, activation=tf.keras.activations.linear, name='reconstructed'),
    ],
    name='decoder'
)

# Reads out the linear dynamics, `K` from the DeepKoopman paper
linear_dynamics = tf.keras.Sequential(
    [
     tf.keras.layers.Dense(LATENT_DIM, activation=tf.keras.activations.linear, name='linear_dynamics_t1')
    ],
    name='linear_dynamics'
)

# Share encoder/decoder for time-points.
# state_input has size (sequence_length, state_dim) to include t=k, t=k+1,...
state_input = tf.keras.Input(shape=input_shape, name='state')

# Unstack state into separate time-points
# Note that axis=0 is the batch dimension
state_sequence = tf.unstack(state_input, axis=1)

# Create the autoencoder graph, which only matters for t=k+0-th time-point
code_sequence = []
for state in state_sequence:
  code = encoder(state)
  code_sequence.append(code)
reconstructed_state_t0 = decoder(code_sequence[0])

# Feed-forward code through linear dynamics
# # Option 1: ignore t=k+0, which is trivially correct
# predicted_code_sequence = []
predicted_code = code_sequence[0]
# Option 2: include t=k+0, which makes the graph a little confusing but might make losses more relevant
predicted_code_sequence = [predicted_code]
for time_offset in range(1, len(code_sequence)):
  predicted_code = linear_dynamics(predicted_code)
  predicted_code_sequence.append(predicted_code)

# Predict/reconstruct future state through the decoder
predicted_state_sequence = [
  decoder(predicted_code) for predicted_code in predicted_code_sequence
]

# Restack predictions across time
codes = tf.stack(code_sequence, axis=1, name='codes')
predicted_codes = tf.stack(predicted_code_sequence, axis=1, name='stack_predicted_codes')
predicted_states = tf.stack(predicted_state_sequence, axis=1, name='stack_predicted_states')


# 1. autoencoder reconstruction loss <- defined in model.compile
# 2. linear dynamics loss <- defined in 
# 3. future state prediction loss
# See "Explicit loss function" in https://www.nature.com/articles/s41467-018-07210-0
# and https://www.nature.com/articles/s41467-018-07210-0/tables/4 for prediction loss parameter
model = tf.keras.Model(
    inputs={'state_input': state_input},
    outputs={'reconstructed_state_t0': reconstructed_state_t0, # for autoencoder loss
             'predicted_codes': predicted_codes, # for linear dynamics
             'predicted_states': predicted_states, # for state prediction loss and autoencoder loss
             },
    name='DeepKoopman'
)

print('model, without loss calculations')
tf.keras.utils.plot_model(model, expand_nested=True)

# Define losses (see "Explicit loss function" section in https://www.nature.com/articles/s41467-018-07210-0)
# Linear dynamics loss: y_{k+1} = K y_k
linear_dynamics_loss = tf.math.reduce_mean(tf.math.squared_difference(codes, predicted_codes))
model.add_loss(linear_dynamics_loss)
# DeepKoopman combines reconstruction (L_recon) and prediction loss (L_pred)
reconstruction_prediction_loss = tf.math.reduce_mean(tf.math.squared_difference(state_input, predicted_states))
model.add_loss(RECON_LOSS_WEIGHT * reconstruction_prediction_loss)
# Note: this example does not yet include L_inf loss and l_2 regularization


# Add metrics
model.add_metric(linear_dynamics_loss, name='linear_dynamics_loss', aggregation='mean')
model.add_metric(reconstruction_prediction_loss, name='reconstruction_prediction_loss', aggregation='mean')

# Full model
print('Full model with losses and metrics')
tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True)

model.compile(optimizer='adam')

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
history = model.fit(
    x={'state_input': x_train},
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data={'state_input': x_test},
    verbose=1,
)

