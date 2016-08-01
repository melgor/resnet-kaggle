import sys
import os

GPU = int(sys.argv[1])

command = "th test_net.lua -v Test_Files/test_256.txt --model {} --GPU {} -save {} -b 16 -aug 1"

for name in sys.argv[2:]:
  command_name = command.format(name, GPU, os.path.basename(name).split(".")[0] + ".h5") 
  print command_name
  os.system(command_name)
