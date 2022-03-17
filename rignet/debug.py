import numpy as np
import os 
import pickle

# single_params = {
#             'shape': shape.detach().cpu().numpy(),  #1,100
#             'exp': exp.detach().cpu().numpy(),      #1,50
#             'pose': pose.detach().cpu().numpy(),    #1,6
#             'cam': cam.detach().cpu().numpy(),      #1,3
#             'verts': trans_vertices.detach().cpu().numpy(),#1,5023,3
#             'albedos':albedos.detach().cpu().numpy(),   # 1,3,256,256
#             'tex': tex.detach().cpu().numpy(),       # 1,50
#             'lit': lights.detach().cpu().numpy()     # 1, ,9, 3
#         }


with open("/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg/flame_p.pickle", 'rb') as f:
    flame_p = pickle.load(f, encoding='latin1')
print (flame_p)

for key in flame_p.keys():
    print ('=-------------')
    print (key, flame_p[key].shape )
    print (flame_p[key])