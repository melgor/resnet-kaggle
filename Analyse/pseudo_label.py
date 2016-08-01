# -*- coding: utf-8 -*-

'''
Label test-data using best model
Then use it for training
'''
import numpy as np
np.random.seed(2016)

import os, sys
import h5py
import shutil

test_data_path = "/media/blcv/drive_2TB/CODE/Kaggle-StateFarm/Data/test_256/"
new_path = "/media/blcv/drive_2TB/CODE/Kaggle-StateFarm/Analyse/data/c{}"
thres_good_example = 0.8  #only use example which confidence is more than thres
thres_ratio = 0.00        #if confidence of predicted class is much more than second one, add it as good example

def parse_csv_file(path):
  with open(path,'r') as f:
    lines = [line.strip().split(",") for line in f]
    
  lines.pop(0) # detele header file
  names = [ line[-1] for line in lines]
  data = np.asarray([ line[:-1] for line in lines], dtype = np.float)
  return data, names

data, names = parse_csv_file(sys.argv[1])
labels = np.argmax(data, axis=1)  
  
num_example  = 0
good_example = 0
for txt,lab in zip(names, labels):
  prob = data[i, lab]
  copy = False
  if prob > thes: 
    good_example += 1
    copy = True
  if prob < thes: 
    #print txt, data[num_example], prob
    sorted_data = np.sort(data[num_example])
    #print sorted_data[-2],sorted_data[-1]
    ratio = sorted_data[-2]/sorted_data[-1]
    if ratio < thres_ratio:
      good_example += 1
      num_example += 1
      copy = True
  if copy:
    new_path_pseudo = new_path.format(str(int(lab)))
    if not os.path.isdir(new_path_pseudo): os.mkdir(new_path_pseudo)
    shutil.copyfile(test_data_path + txt, os.path.join(new_path_pseudo, txt))
  num_example += 1
  

print "Good example: ", good_example

