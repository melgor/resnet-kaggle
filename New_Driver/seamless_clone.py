# Standard imports
import cv2
import sys
import numpy as np 
 
# Read images
src = cv2.imread(sys.argv[1])
dst = cv2.imread(sys.argv[2])
 
 
# Create a rough mask around the airplane.
src_mask = np.zeros(src.shape, src.dtype)
src_mask[16:190,134:310] = 1.0
#poly = np.array([ [16,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
#cv2.fillPoly(src_mask, [poly], (255, 255, 255))
 
# This is where the CENTER of the airplane will be placed
center = (0,0)
 
# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
 
# Save result
cv2.imwrite("opencv-seamless-cloning-example.jpg", output);
