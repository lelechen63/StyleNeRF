import os
from argparse import ArgumentParser
from collections import OrderedDict
import torch
import torch.nn as nn
import random
import pickle
from options.train_options import TrainOptions
import numpy as np
import sys
# define flame config

def dict2obj(d):
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o

opt = TrainOptions().parse()


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
        'image_size': opt.imgsize,
        'e_lr': 0.005,
        'e_wd': 0.0001,
        'savefolder': '/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg',
        # weights of losses and reg terms
        'w_pho': 100,
        'w_lmks': 0,
        'w_shape_reg': 1e-4,
        'w_expr_reg': 1e-4,
        'w_albedo_reg':1e-4,
        'w_lit_reg':1e-4,
        'w_pose_reg': 0,
    }

flame_config = dict2obj(flame_config)

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

if opt.isTrain:
    print ( opt.name)
    model.train()
else:
    if opt.name == 'Latent2Code':
        model.test()
    elif  opt.name == 'rig':
        model.test()
        