#!/usr/bin/env python
import numpy as np
import argparse,os
from sklearn.metrics import confusion_matrix, accuracy_score
from extractor import Prediction
from utils import *


parser = argparse.ArgumentParser(description='Run Caffe model from dir and given label')
parser.add_argument('--images', required=True,help='path to file with paths to images')
parser.add_argument('--proto', required=True,help='path to proto file')
parser.add_argument('--bin', required=True,help='path to binary Net')
parser.add_argument('--folder', required=True,help='folder where save featues')


def extract_multi(args):
  # Extract  Featues from Net using prot_file from images from input file
  # it will save patch of max_value features
  pred = Prediction(args.proto,args.bin)
  
  max_value = 16 
  curr_value = 0
  curr_value_all = 0
  list_all_result = list()
  list_good_class_all = list()
  list_all_paths = list()
  create_dir(args.folder)
  with open(args.images,'r') as file_image:
    list_images = list()
    list_good_class = list()
    for idx,line in enumerate(file_image):
      splitted = line.strip().split(" ")
      #for i in range(10): #add for each oversample example
      list_good_class_all.append(int(splitted[1]))
      list_images.append(splitted[0])
      curr_value = curr_value + 1
      if curr_value < max_value:
        continue
      else:
        #predict using value
        predictions = pred.predict_multi(list_images)
        list_all_result.append(predictions)
        list_all_paths.extend(list_images)
        list_images = list()
        curr_value = 0
        curr_value_all += max_value
        print "Predicted:", curr_value_all
    
    
    #predict last package of data, which is smaller than max_value
    if len(list_images) > 0:
      predictions = pred.predict_multi(list_images)
      list_all_result.append(predictions)
      list_all_paths.extend(list_images)

  data = np.vstack(list_all_result)   
  main_name = "{}/extracted_feature_kaggle_224.{}"
  save_H5PY(main_name.format(args.folder,'h5py'), data, np.asarray(list_good_class_all))
  #save_pred_paths_H5PY(main_name.format(args.folder,'h5py'),
                    #data,np.asarray(list_good_class_all),np.asarray(list_all_paths))
  #toCsv(args.folder, data)
  #test set
  y_truth = np.asarray(list_good_class_all)
  y_pred = np.vstack(list_all_result)
  y_pred_label = np.argmax(y_pred, axis=1)
  print y_truth, y_pred_label
  print "Accuracy: ", accuracy_score(y_truth,y_pred_label)
  
  
if __name__ == '__main__':
  args        = parser.parse_args()
  extract_multi(args)
  
    