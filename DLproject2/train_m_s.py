from __future__ import division, print_function, absolute_import
import h5py
from lib.config import params_setup
from lib.googlenet import GoogLeNet
import os, tflearn
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from lib import data_util

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist



h5f_train = h5py.File('dataset_train_s.h5', 'r')
h5f_test = h5py.File('dataset_test_s.h5', 'r')
h5f_val = h5py.File('dataset_val_s.h5', 'r')
X = h5f_train['X']
Y = h5f_train['Y']
X_test = h5f_test['X']
X_val= h5f_val['X']
Y_val = h5f_val['Y']

img_size = 32
label_size = 5
model_name = 'flowerClassify_s'
model_path = 'models/%s' % model_name
if not os.path.exists(model_path): os.makedirs(model_path)
def get_checkpoint(model_path):
    ckpt_path = '%s/checkpoint' % model_path
    if not os.path.exists(ckpt_path): return
    with open(ckpt_path, 'r') as f:
        lines = [line.split(':')[1].replace('\"', '').strip() for line in f.readlines()]
        lines = lines[1:][::-1]
    for line in lines:
        path = "%s/%s" % (model_path, line)
        if os.path.exists(path):
            return path
ckpt = get_checkpoint(model_path)
if ckpt:
    model.load(ckpt)
    print("load existing checkpoint from %s" % ckpt)



network = input_data(shape=[None, 32, 32, 3], name='input')
network = conv_2d(network, 32, 3, strides=2, activation='relu')
network = conv_2d(network, 64, 5, activation='relu')
network = conv_2d(network, 32, 5, activation='relu')
network = conv_2d(network, 32, 3, strides=2, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='sgd', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')



model = tflearn.DNN(network, checkpoint_path=model_path+'/ckpt',
best_checkpoint_path=model_path+'/model',max_checkpoints=3, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=500,
           validation_set=({'input': X_val}, {'target': Y_val}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')

model.save('mymodel_s.tflearn')
model.load('mymodel_s.tflearn')
predictions = model.predict_label(X_test)
new_y = np.argmax(predictions, axis=1)
np.savetxt('prediction_s.txt',new_y,fmt="%d",newline='\n')

# load existing model checkpoint
