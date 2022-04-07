import os
import numpy as np
import pickle

with open("/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg/flame_p.pickle", 'rb') as f:
    flame_p = pickle.load(f, encoding='latin1')
print (flame_p)