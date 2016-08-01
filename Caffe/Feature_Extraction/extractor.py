import numpy as np
import cv2
import os
import sys
# Make sure that caffe is on the python path:
# this file is expected to be in {caffe_root}/examples
caffe_root = '/home/blcv/LIB/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

class Prediction(object):
  """docstring for Prediction_Normalme"""
  def __init__(self, proto_path,bin_path):
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = proto_path 
    PRETRAINED = bin_path 
    caffe.set_mode_gpu()
    caffe.set_device(0)
    #caffe.set_mode_cpu()
    print "Create Data"
    self.net = caffe.Classifier (MODEL_FILE,PRETRAINED,
              mean = np.array([104,117,123]),
              raw_scale = 255,
              image_dims = (224, 300),
              channel_swap = (2,1,0)
              )        

    
  def predict(self, image):
    """Predict using Caffe normal model"""
    input_image = caffe.io.load_image(image,color=False)
    prediction = self.net.predict([input_image], oversample=False)
    return prediction
      
  def predict_multi(self, images):
    """Predict using Caffe normal model"""
    list_input = list()
    for image in images: 
      try:
        list_input.append(caffe.io.load_image(image,color=True))
      except:
        print image
      # image = image.replace(".png",".jpg")
      # list_input.append(caffe.io.load_image(image,color=True))
    prediction = self.net.predict(list_input, oversample=False)
    return prediction
  