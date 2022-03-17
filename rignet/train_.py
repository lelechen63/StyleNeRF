import os
from argparse import ArgumentParser
from collections import OrderedDict
import torch
import torch.nn as nn
import random
import pickle
import pytorch_lightning as pl
from options.train_options import TrainOptions
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import sys
sys.path.append('./photometric_optimization')
import util
# define flame config
flame_config = {
        # FLAME
        'flame_model_path': '/home/uss00022/lelechen/basic/flame_data/data/generic_model.pkl',  # acquire it from FLAME project page
        'flame_lmk_embedding_path': '/home/uss00022/lelechen/basic/flame_data/data/landmark_embedding.npy',
        'tex_space_path': '/home/uss00022/lelechen/basic/flame_data/data/FLAME_texture.npz',  # acquire it from FLAME project page
        'camera_params': 3,
        'shape_params': 100,
        'expression_params': 50,
        'pose_params': 6,
        'tex_params': 50,
        'use_face_contour': True,

        'batch_size': 1,
        'image_size': 512,
        'e_lr': 0.005,
        'e_wd': 0.0001,
        'savefolder': '/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg',
        # weights of losses and reg terms
        'w_pho': 8,
        'w_lmks': 100,
        'w_shape_reg': 1e-4,
        'w_expr_reg': 1e-4,
        'w_pose_reg': 0,
    }

flame_config = util.dict2obj(flame_config)

opt = TrainOptions().parse()

# if opt.debug:
#     opt.nThreads = 1

opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if  opt.name == 'Latent2Code':
    from latent2code import Latent2CodeModule as module
    model = module(flame_config, opt )
elif opt.name =='rig':
    from rig import RigModule as module
    model = module(flame_config, opt)

print (opt.isTrain,'!!!!!')
if opt.isTrain:
    print ( opt.name)
    model.train()
    print ('+++++++++')
else:
    print ('!!!!!!' + opt.name + '!!!!!!!!')
    if opt.name == 'Latent2Code':
        model.debug()
        