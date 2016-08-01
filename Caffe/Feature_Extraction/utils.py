# -*- coding: utf-8 -*-
# @Author: melgor
# @Date:   2014-11-27 11:54:32
# @Last Modified 2015-02-27
# @Last Modified time: 2016-05-22 12:09:15
import numpy as np
import json
import cPickle
import os
import gzip
import h5py
import csv
from collections import namedtuple
from sklearn.preprocessing import Normalizer

Feature = namedtuple('Feature', 'data label')

#TODO add compressing using: Return 2x times smaller
#import bz2
        #import cPickle as pickle
        #with bz2.BZ2File('test.pbz2', 'w') as f:
            #pickle.dump(l, f)
def load_cPickle(name_file):
  '''Load file in cPickle format'''
  f = gzip.open(name_file,'rb')
  tmp = cPickle.load(f)
  f.close()
  return tmp

def save_cPickle(name_file, data ):
  '''Save file in cPickle format, delete if exist'''
  if os.path.isfile(name_file):
      os.remove(name_file)
  f = gzip.open(name_file,'wb')
  cPickle.dump(data,f,protocol=2)
  f.close()
  
  
def save_H5PY(name_file, data, label):
  '''Save data as H5PY format'''
  if os.path.isfile(name_file):
      os.remove(name_file)
  # HDF5 is pretty efficient, but can be further compressed.
  comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
  with h5py.File(name_file, 'w') as f:
      f.create_dataset('prediction', data=data, **comp_kwargs)
      f.create_dataset('label', data=label.astype(np.float32), **comp_kwargs)


'''Save data, labels and paths to images as H5PY format'''
def save_pred_paths_H5PY(name_file, data, label, paths, ):
  if os.path.isfile(name_file):
      os.remove(name_file)
  # HDF5 is pretty efficient, but can be further compressed.
  comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
  with h5py.File(name_file, 'w') as f:
      f.create_dataset('data', data=data, **comp_kwargs)
      f.create_dataset('label', data=label, **comp_kwargs)
      f.create_dataset('paths', data=paths, **comp_kwargs)
      
  #with open(os.path.join(dirname, 'train.txt'), 'a') as f:
      #f.write(val_filename + '\n')
      
def load_H5PY(name_file):
    '''Read data from H5PY format'''
    with h5py.File(name_file,  "r") as f:
      feature = Feature(f['data'].value,f['label'].value)
    
    return feature

def create_dir(path):
  print path
  if not os.path.isdir(path):
    os.makedirs(path)