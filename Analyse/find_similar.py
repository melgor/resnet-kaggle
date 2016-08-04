import os,sys
import shutil
import h5py 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm

def load_H5PY(name_file):
    '''Read data from H5PY format'''
    with h5py.File(name_file,  "r") as f:
      feature,paths = f['data'].value,f['paths'].value
    return feature, paths
  
  

feature, paths = load_H5PY(sys.argv[1])
feature = feature[:,:,0,0]
dest_folder = os.path.basename(sys.argv[1]).split(".")[0].split("_")[-1]
os.mkdir(dest_folder)

alg = KMeans(init='k-means++', n_clusters=28, n_init=10,  n_jobs=8)
labels = alg.fit_predict(feature)

for label,path in zip(labels, paths):
  des = os.path.join(dest_folder,str(label))
  if not os.path.isdir(des): os.mkdir(des)
  shutil.copyfile(path, os.path.join(des, os.path.basename(path)))
  
#distance_matrix = pairwise_distances(feature)
#print "Matrix", feature.shape
#ax = plt.subplot(1, 1, 1)
#plt.imshow(np.abs(distance_matrix), interpolation='nearest')#, interpolation='nearest' )
#ax.set_aspect('equal')
#plt.colorbar(orientation='vertical')
#plt.show()  