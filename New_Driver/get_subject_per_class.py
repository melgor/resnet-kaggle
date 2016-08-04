import os,sys
import shutil

main_folder = "/media/blcv/drive_2TB/CODE/Kaggle-StateFarm/Data/train/train_org/"

def mkdir(path):
  p  = os.path.dirname(path)
  if os.path.isdir(p): return
  os.makedirs(p)

with open(sys.argv[1], 'r') as f:
  data = [line.strip().split(",") for line in f]
  
  
data.pop(0)
for exp in data:
  path = os.path.join(exp[0], exp[1],exp[2])
  path_main = os.path.join( exp[1],exp[2])
  mkdir(path)
  shutil.copy( main_folder + path_main, path)
