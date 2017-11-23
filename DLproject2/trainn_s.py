from __future__ import division, print_function, absolute_import
from lib import data_util
from lib.config import params_setup
from lib.googlenet import GoogLeNet
from datetime import datetime
import pickle, gzip
import numpy as np
#from tflearn.data_utils import build_hdf5_image_dataset
import os, random, pickle, gzip
import numpy as np
from PIL import Image
from datetime import datetime
import h5py
from lib.data_util import load_image
from lib.data_util import resize_image
from lib.data_util import convert_color
from lib.data_util import pil_to_nparray
from lib.data_util import to_categorical
from sklearn import preprocessing




target_train_path='/Users/sky/Documents/GitHub/imgrec/data/train.txt'
target_val_path = '/Users/sky/Documents/GitHub/imgrec/data/val.txt'
target_test_path = '/Users/sky/Documents/GitHub/imgrec/data/test.txt'
image_shape=(32,32)
output_train_path = 'dataset_train_s.h5'
output_test_path = 'dataset_test_s.h5'
output_val_path = 'dataset_val_s.h5'
mode='file'
categorical_labels=False
normalize=True
grayscale=False
files_extension=None
chunks=True




def build_hdf5_image_dataset(target_path, image_shape, output_path,
                             mode, categorical_labels,
                             normalize, grayscale,
                             files_extension, chunks):
    enc = preprocessing.OneHotEncoder()
    enc.fit([[1.],[0.],[2.],[3.],[4.]])

    with open(target_path, 'r') as f:
        images, labels = [], []
        lines = f.readlines()
        for l in lines:
            l = l.strip('\n').split()
            images.append(l[0])
            labels.append(int(l[1]))
    n_classes = np.max(labels) + 1

    if not grayscale:
        d_imgshape = (len(images), image_shape[0], image_shape[1], 3)
    else:
        d_imgshape = (len(images), image_shape[0], image_shape[1])
    d_labelshape = (len(images),n_classes)

    dataset = h5py.File(output_path, 'w')
    dataset.create_dataset('X', d_imgshape, chunks=chunks)
    dataset.create_dataset('Y', d_labelshape, chunks=chunks)

    for i in range(len(images)):
        img = load_image(images[i])
        width, height = img.size
        if width != image_shape[0] or height != image_shape[1]:
            img = resize_image(img, image_shape[0], image_shape[1])
        if grayscale:
            img = convert_color(img, 'L')
        elif img.mode == 'L':
            img = convert_color(img, 'RGB')

        img = pil_to_nparray(img)
        if normalize:
            img /= 255.

        dataset['X'][i] = img
        #if categorical_labels:
            #dataset['Y'][i] = to_categorical([labels[i]], n_classes)[0]
        #else:
        labels[i]=enc.transform([[labels[i]]]).toarray()
        dataset['Y'][i] = labels[i]

def build_hdf5_image_dataset_test(target_path, image_shape, output_path,
                             mode, categorical_labels,
                             normalize, grayscale,
                             files_extension, chunks):
    with open(target_path, 'r') as f:
        images = []
        lines = f.readlines()
        for l in lines:
            l = l.strip('\n').split()
            images.append(l[0])

    if not grayscale:
        d_imgshape = (len(images), image_shape[0], image_shape[1], 3)
    else:
        d_imgshape = (len(images), image_shape[0], image_shape[1])

    dataset = h5py.File(output_path, 'w')
    dataset.create_dataset('X', d_imgshape, chunks=chunks)

    for i in range(len(images)):
        img = load_image(images[i])
        width, height = img.size
        if width != image_shape[0] or height != image_shape[1]:
            img = resize_image(img, image_shape[0], image_shape[1])
        if grayscale:
            img = convert_color(img, 'L')
        elif img.mode == 'L':
            img = convert_color(img, 'RGB')

        img = pil_to_nparray(img)
        if normalize:
            img /= 255.
        dataset['X'][i] = img


dataset_train = build_hdf5_image_dataset(target_train_path, image_shape, output_train_path,
                             mode, categorical_labels,
                             normalize, grayscale,
                             files_extension, chunks)

dataset_val = build_hdf5_image_dataset(target_val_path, image_shape, output_val_path,
                             mode, categorical_labels,
                             normalize, grayscale,
                             files_extension, chunks)

dataset_test = build_hdf5_image_dataset_test(target_test_path, image_shape, output_test_path,
                             mode, categorical_labels,
                             normalize, grayscale,
                             files_extension, chunks)
