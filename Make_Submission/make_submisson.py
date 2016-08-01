# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(2016)
import os, sys
import pickle
import datetime
import pandas as pd
import h5py

'''
Create Submission for given predicitions and path to images
argv[1]: paths to images, order same like in predictions
argv[2]: name of submission
argv[3:]: list of predicions which will be used for creating the submission 
'''

def create_submission(predictions, test_id, name_submission):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = name_submission + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    print sub_file
    result1.to_csv(sub_file, index=False)
    
name_submission = sys.argv[2] 
with open(sys.argv[1],'r') as f:
  test_id = [os.path.basename(line.strip().split(" ")[0]) for line in f]
  
list_pred = list()               
for pred in sys.argv[3:]:
  with h5py.File(pred, 'r') as f:
    data = f['prediction'].value
    list_pred.append(data)

# Sum the prediciton and divide by the quantity of them
data = np.zeros(list_pred[0].shape)

for d in list_pred:
  print d.mean(), d.shape
  data += d

data /= float(len(list_pred))

print data.shape

create_submission(data, test_id, name_submission)
