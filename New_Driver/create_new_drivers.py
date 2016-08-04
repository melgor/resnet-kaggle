# Standard imports
import cv2
import sys, os
import numpy as np
import json
import random
from joblib import Parallel, delayed

targets = ['c1','c2','c3','c4','c5','c6','c7','c8','c9']#'c0'

def make_dir(path):
  if not os.path.isdir(path): os.makedirs(path)
  
input_drivers = sys.argv[1:]
print input_drivers

def get_images(target):
  # Gather faces
  faces_driver = dict()
  for dr in input_drivers:
    with open(os.path.join(dr,'{}.json'.format(target))) as f:
      data = json.load(f)
    faces_driver[dr] = list()
    for key,value in data.iteritems():
      img = cv2.imread(os.path.join(dr,key))
      #face = img[value[1]:value[3], value[0]: value[2]]
      faces_driver[dr].append((os.path.join(dr,key),value))


  # Create new driver:
  images = 2000
  curr_img = 0
  for i in range(3000):
    image_key = random.choice(faces_driver.keys())
    while True:
      face_key = random.choice(faces_driver.keys())
      if face_key != image_key: break
    print face_key, image_key
    # Get base image and center of face
    (path_image_base, rect) = random.choice(faces_driver[image_key])
    #print path_image_base, rect
    image = cv2.imread(path_image_base)
    center = (int((rect[0]+ rect[2])/2.0), int((rect[1]+ rect[3])/2.0))
	      
    # Get new face
    (path_image_face, rect) = random.choice(faces_driver[face_key])     
    #print path_image_face, rect
    face = cv2.imread(path_image_face)
    face = face[int(rect[1]):int(rect[3]), int(rect[0]): int(rect[2])]
    src_mask = np.ones(face.shape, image.dtype)
    src_mask[:,:] = 255.0
    #print face.shape, face.dtype, image.shape, image.dtype, src_mask.shape, src_mask.dtype,center
    # Clone seamlessly.
    try:
      output = cv2.seamlessClone(face, image, src_mask, center, cv2.NORMAL_CLONE)
      new_name = "new_driver/{}_{}/{}/".format(path_image_base[:4],path_image_face[:4],target)
      make_dir(new_name)
      cv2.imwrite(os.path.join( new_name, str(curr_img) + ".jpg"), output)
      #cv2.imwrite(str(curr_img) + ".jpg", output)
      curr_img += 1
    except:
      print 'error'
      pass
    
    print i
    if curr_img == images: break
  
Parallel(n_jobs=10, verbose=10)(delayed(get_images)(f) for f in targets)