
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from skimage.transform import resize
import face_alignment

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import datetime
import imageio
from PIL import Image
import cv2
from model import BiSeNet

import sys
sys.path.append('./models/')
from FLAME import FLAME, FLAMETex
from Flamerenderer import FlameRenderer as Renderer
import util
torch.backends.cudnn.benchmark = True
sys.path.append('/home/uss00022/lelechen/github/CIPS-3D/utils')

import tensor_util
import argparse

import datetime
import pathlib
import os
import pickle
#----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--k",
                     type=int,
                     default=0)
    parser.add_argument("--imgsize",
                     type=int,
                     default=256)
    return parser.parse_args()

parse = parse_args()

    
config = {
        # FLAME
        "savefolder" : '/nfs/STG/CodecAvatar/lelechen/FFHQ/generated_stylenerf/flame2/',
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
        'image_size': parse.imgsize,
        'e_lr': 0.005,
        'e_wd': 0.0001,
        'w_pho': 8,
        'w_lmks': 1,
        'w_shape_reg': 1e-4,
        'w_expr_reg': 1e-4,
        'w_pose_reg': 0,
    }

config = util.dict2obj(config)

def vis_tensor(image_tensor = None, image_path = None, land_tensor = None, cam = None,  visind =0, device = torch.device("cuda")):
    if land_tensor is not None:
        lmark = util.batch_orth_proj(land_tensor.to(device), cam.to(device))
        lmark[..., 1:] = - lmark[..., 1:]
        lmark = util.tensor_vis_landmarks(image_tensor.to(device)[visind].unsqueeze(0),lmark[visind].unsqueeze(0))
        output = lmark.squeeze(0)
    else:
        output = image_tensor.data[visind].detach().cpu() #  * self.stdtex + self.meantex 
    output = tensor_util.tensor2im(output  , normalize = False)
    output = np.ascontiguousarray(output, dtype=np.uint8)
    output = util.writeText(output, image_path)
    output = np.ascontiguousarray(output, dtype=np.uint8)
    output = np.clip(output, 0, 255)

    return output
device = torch.device("cuda")
config.batch_size = 1
config.image_size = parse.imgsize
flame = FLAME(config).to(device)
flametex = FLAMETex(config).to(device)
mesh_file = '/home/uss00022/lelechen/basic/flame_data/data/head_template_mesh.obj'
render = Renderer(config.image_size, obj_filename=mesh_file).to(device)

def output( p, idx, config = config, flame = flame, flametex = flametex, render = render, device = torch.device("cuda")  ):
    shape, exp,tex, lit, pose, cam = p[0], p[1],p[2],p[3],p[4],p[5]

    root = '/nfs/STG/CodecAvatar/lelechen/FFHQ/generated_stylenerf'

    shape = torch.FloatTensor(shape).to(device)
    exp = torch.FloatTensor(exp).to(device)
    pose = torch.FloatTensor(pose).to(device)
    cam = torch.FloatTensor(cam).to(device)
    tex = torch.FloatTensor(tex).to(device)
    lights = torch.FloatTensor(lit).to(device)

    vertices, landmarks2d, landmarks3d_save = flame(shape_params=shape, expression_params=exp, pose_params=pose)
    trans_vertices = util.batch_orth_proj(vertices, cam)
    trans_vertices[..., 1:] = - trans_vertices[..., 1:]
    landmarks2d = util.batch_orth_proj(landmarks2d, cam)
    landmarks2d[..., 1:] = - landmarks2d[..., 1:]
    landmarks3d = util.batch_orth_proj(landmarks3d_save, cam)
    landmarks3d[..., 1:] = - landmarks3d[..., 1:]
    ## render
    albedos = flametex(tex, config.image_size ) / 255.
    ops = render(vertices, trans_vertices, albedos, lights)
    predicted_images = ops['images']

    genimage = vis_tensor(image_tensor= predicted_images, 
            image_path = str(idx),
            device = device
            )
    genimage = cv2.cvtColor(genimage, cv2.COLOR_RGB2BGR)

    return genimage

def varify( parse = parse):
    device ='cuda'
    k =  parse.k
    root = '/nfs/STG/CodecAvatar/lelechen/FFHQ/generated_stylenerf'
    
    idx1 = 11
    img_p1 = os.path.join( root, 'images', '%06d.png'%idx1)
    flame_path = config.savefolder + '/%06d/flame_p.pickle'%idx1
    with open(flame_path, 'rb') as f:
        flame_p = pickle.load(f, encoding='latin1')
    p1 = [flame_p['shape'], flame_p['exp'],flame_p['tex'], flame_p['lit'], flame_p['pose'], flame_p['cam']]

    idx2 = 8
    img_p2 = os.path.join( root, 'images', '%06d.png'%idx2)
    flame_path = config.savefolder + '/%06d/flame_p.pickle'%idx2
    with open(flame_path, 'rb') as f:
        flame_p = pickle.load(f, encoding='latin1')
    p2 = [flame_p['shape'], flame_p['exp'], flame_p['tex'], flame_p['lit'], flame_p['pose'], flame_p['cam']]

    ps = []

    # ps[0], all are p1, but ps[90][0] is p2[0], etc.
    for i in range(4):
        ps.append([])
        for j in range(6):
            if j ==i:
                ps[i].append(p2[j])
            else:
                ps[i].append(p1[j])
    gtimage1 = cv2.imread(img_p1)
    gtimage1 = cv2.resize(gtimage1, (config.image_size,config.image_size), interpolation = cv2.INTER_AREA)
    
    gtimage2 = cv2.imread(img_p2)
    gtimage2 = cv2.resize(gtimage2, (config.image_size,config.image_size), interpolation = cv2.INTER_AREA)


    change_chart =['shape', 'exp', 'tex', 'lit']    

    genimage1 = output(p1,'p1' )
    genimage2 = output(p2,'p2' )
    ff = cv2.hconcat([genimage1, genimage1, genimage2])
    
    for i,p in enumerate(ps):
        genimage = output(p,change_chart[i] )
        img = cv2.hconcat([genimage, gtimage1, gtimage2])
        ff = cv2.vconcat([ff, img])

    cv2.imwrite('gg.png', ff)


varify()