# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:04:02 2022

@author: gibson48
"""

import os
import json

import tensorflow as tf
import mnist_setup

#batch_size = 64
#single_worker_dataset = mnist_setup.mnist_dataset(batch_size)
#single_worker_model = mnist_setup.build_and_compile_cnn_model()
#single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)

tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}

os.environ['TF_CONFIG'] = json.dumps(tf_config)



per_worker_batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_setup.mnist_dataset(global_batch_size)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = mnist_setup.build_and_compile_cnn_model()


multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)
