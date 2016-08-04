import cv2
import numpy as np
import sys, os
from matplotlib import pyplot as plt
import glob, json
#img = cv2.imread(sys.argv[1],0)
#img2 = img.copy()
#template = cv2.imread(sys.argv[2],0)


def detect(img2, template):
  w, h = template.shape[::-1]

  # All the 6 methods for comparison in a list
  methods = [ 'cv2.TM_CCOEFF_NORMED',
              'cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF_NORMED']


  rectangles = list()
  for meth in methods:
      img = img2.copy()
      method = eval(meth)

      # Apply template Matching
      res = cv2.matchTemplate(img,template,method)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

      # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
      if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
          top_left = min_loc
      else:
          top_left = max_loc
      bottom_right = (top_left[0] + w, top_left[1] + h)
      rectangles.append((top_left[0],top_left[1],top_left[0] + w,top_left[1] + h))
      
  rect = np.mean(np.asarray(rectangles, dtype = np.int),axis=0)    
  return rect    
  #cv2.rectangle(img,(int(rect[0]),int(rect[1])), (int(rect[2]),int(rect[3])), 255, 2)

  #plt.imshow(img,cmap = 'gray')
  #plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
  #plt.show()
  
imgs = glob.glob("{}/*.jpg".format(sys.argv[2]))
template  =  cv2.imread(sys.argv[1],0)
results   = dict()
for imgpath in imgs:
  img = cv2.imread(imgpath,0)
  rect = detect(img, template)
  results[imgpath] = rect.tolist()
  #cv2.rectangle(img,(int(rect[0]),int(rect[1])), (int(rect[2]),int(rect[3])), 255, 2)
  #cv2.imwrite(os.path.basename(imgpath),img)
  
with open(os.path.dirname(sys.argv[2]) + ".json",'w') as f:
  json.dump(results,f)