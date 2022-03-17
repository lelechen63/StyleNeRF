import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import torch as th
import torch.nn as nn
import functools
import torchvision
from collections import OrderedDict
import os
from os import path as osp
import numpy as np
import pickle
from PIL import Image
import cv2
import sys
sys.path.append('./photometric_optimization/')
from renderer import Renderer
import util
from models.FLAME import FLAME, FLAMETex
sys.path.append('/home/uss00022/lelechen/github/CIPS-3D/utils')
from visualizer import Visualizer
import tensor_util
from blocks import *
import face_alignment

class Latent2Code(nn.Module):
    def __init__(self, flame_config, opt ):
        super().__init__()
        self.opt = opt
        # self.save_hyperparameters()
        self.flame_config = flame_config
    
        self.image_size = self.flame_config.image_size
        # networks
        self.nerf_latent_dim = 256
        self.gan_latent_dim = 512
        self.shape_dim = 100
        self.exp_dim = 50
        self.albedo_dim = 50
        self.lit_dim = 27
        self.Latent2ShapeExpCode = self.build_Latent2ShapeExpCodeFea( weight = '' if opt.isTrain else opt.Latent2ShapeExpCode_weight)
        self.latent2shape = self.build_latent2shape( weight = '' if opt.isTrain else opt.latent2shape_weight)
        self.latent2exp = self.build_latent2exp(weight = '' if opt.isTrain else opt.latent2exp_weight)
        self.Latent2AlbedoLitCode = self.build_Latent2AlbedoLitCodeFea(weight = '' if opt.isTrain else opt.Latent2AlbedoLitCode_weight)
        self.latent2albedo = self.build_latent2albedo(weight = '' if opt.isTrain else opt.latent2albedo_weight)
        self.latent2lit = self.build_latent2lit(weight = '' if opt.isTrain else opt.latent2lit_weight)
        if opt.isTrain:
            self._initialize_weights()
        self.flame = FLAME(self.flame_config).to('cuda')
        self.flametex = FLAMETex(self.flame_config).to('cuda')
        self._setup_renderer()

        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.ckpt_path, exist_ok = True)
    
    def build_Latent2ShapeExpCodeFea(self, weight = ''):
        Latent2ShapeExpCode = th.nn.Sequential(
            LinearWN( self.nerf_latent_dim , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for latent2ShapeExpCode feature extraction network')
            Latent2ShapeExpCode.load_state_dict(torch.load(weight))
        return Latent2ShapeExpCode
    def build_latent2shape(self,  weight = ''):
        latent2shape= th.nn.Sequential(
            LinearWN( 256 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.shape_dim )
        )
        if len(weight) > 0:
            print ('loading weights for latent2Shape network')
            latent2shape.load_state_dict(torch.load(weight))
        return latent2shape
    def build_latent2exp(self, weight = ''):
        latent2exp= th.nn.Sequential(
            LinearWN( 256 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.exp_dim )
        )
        if len(weight) > 0:
            print ('loading weights for latent2exp network')
            latent2exp.load_state_dict(torch.load(weight))
        return latent2exp
    def build_Latent2AlbedoLitCodeFea(self, weight = ''):
        Latent2AlbedoLitCode = th.nn.Sequential(
            LinearWN( self.gan_latent_dim , 512 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 512, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for Latent2AlbedoLitCode feature extraction network')
            Latent2AlbedoLitCode.load_state_dict(torch.load(weight))
        return Latent2AlbedoLitCode
    
    def build_latent2albedo(self, weight = ''):
        latent2albedo= th.nn.Sequential(
            LinearWN( 256 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.albedo_dim )
        )
        if len(weight) > 0:
            print ('loading weights for latent2albedo feature extraction network')
            latent2albedo.load_state_dict(torch.load(weight))
        return latent2albedo
    def build_latent2lit(self, weight = ''):
        latent2lit= th.nn.Sequential(
            LinearWN( 256 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.lit_dim )
        )
        if len(weight) > 0:
            print ('loading weights for latent2lit feature extraction network')
            latent2lit.load_state_dict(torch.load(weight))
        return latent2lit

    def _setup_renderer(self):
        mesh_file = '/home/uss00022/lelechen/basic/flame_data/data/head_template_mesh.obj'
        self.render = Renderer(self.image_size, obj_filename=mesh_file).to('cuda')
    
    def forward(self, shape_latent, appearance_latent, cam, pose, flameshape = None, flameexp= None, flametex= None, flamelit= None ):
        
        shape_fea = self.Latent2ShapeExpCode(shape_latent)
        shapecode = self.latent2shape(shape_fea)
        expcode = self.latent2exp(shape_fea)
        
        app_fea = self.Latent2AlbedoLitCode(appearance_latent)
        albedocode = self.latent2albedo(app_fea)
        litcode = self.latent2lit(app_fea)
        
        return_list = {}
        if self.opt.supervision =='render' or flameshape != None:
            vertices, landmarks2d, landmarks3d = self.flame(shape_params=shapecode, expression_params=expcode, pose_params=pose)
            trans_vertices = util.batch_orth_proj(vertices, cam)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]

            ## render
            albedos = self.flametex(albedocode, self.image_size) / 255.
            ops = self.render(vertices, trans_vertices, albedos, litcode.view(shape_latent.shape[0], 9,3))
            predicted_images = ops['images']

            return_list['landmarks3d'] = landmarks3d
            return_list['predicted_images'] = predicted_images
        else:
            return_list['expcode'] = expcode
            return_list['shapecode'] = shapecode
            return_list['litcode'] = litcode
            return_list['albedocode'] = albedocode
            
        if flameshape != None:
            flamelit = flamelit.view(-1, 9,3)        
            recons_vertices, _, recons_landmarks3d = self.flame(shape_params=flameshape, expression_params=flameexp, pose_params=pose)
            recons_trans_vertices = util.batch_orth_proj(recons_vertices, cam)
            recons_trans_vertices[..., 1:] = - recons_trans_vertices[..., 1:]

            ## render
            recons_albedos = self.flametex(flametex, self.image_size) / 255.
            recons_ops = self.render(recons_vertices, recons_trans_vertices, recons_albedos, flamelit)
            recons_images = recons_ops['images']
            return_list['recons_images'] = recons_images
            
        
        return return_list
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Latent2Code2(nn.Module):
    def __init__(self, flame_config, opt ):
        super().__init__()
        self.opt = opt
        # self.save_hyperparameters()
        self.flame_config = flame_config
    
        self.image_size = self.flame_config.image_size
        # networks
        self.latent_dim = 512 * 17
        self.shape_dim = 100
        self.exp_dim = 50
        self.albedo_dim = 50
        self.lit_dim = 27
        self.Latent2fea = self.build_Latent2CodeFea( weight = '' if opt.isTrain else opt.Latent2ShapeExpCode_weight)
        self.latent2shape = self.build_latent2shape( weight = '' if opt.isTrain else opt.latent2shape_weight)
        self.latent2exp = self.build_latent2exp(weight = '' if opt.isTrain else opt.latent2exp_weight)
        self.latent2albedo = self.build_latent2albedo(weight = '' if opt.isTrain else opt.latent2albedo_weight)
        self.latent2lit = self.build_latent2lit(weight = '' if opt.isTrain else opt.latent2lit_weight)
        if opt.isTrain:
            self._initialize_weights()
        self.flame = FLAME(self.flame_config).to('cuda')
        self.flametex = FLAMETex(self.flame_config).to('cuda')
        self._setup_renderer()

        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.ckpt_path, exist_ok = True)
    
    def build_Latent2CodeFea(self, weight = ''):
        Latent2ShapeExpCode = th.nn.Sequential(
            LinearWN( self.latent_dim , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for latent2ShapeExpCode feature extraction network')
            Latent2ShapeExpCode.load_state_dict(torch.load(weight))
        return Latent2ShapeExpCode
    def build_latent2shape(self,  weight = ''):
        latent2shape= th.nn.Sequential(
            LinearWN( 256 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.shape_dim )
        )
        if len(weight) > 0:
            print ('loading weights for latent2Shape network')
            latent2shape.load_state_dict(torch.load(weight))
        return latent2shape
    def build_latent2exp(self, weight = ''):
        latent2exp= th.nn.Sequential(
            LinearWN( 256 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.exp_dim )
        )
        if len(weight) > 0:
            print ('loading weights for latent2exp network')
            latent2exp.load_state_dict(torch.load(weight))
        return latent2exp

    def build_latent2albedo(self, weight = ''):
        latent2albedo= th.nn.Sequential(
            LinearWN( 256 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.albedo_dim )
        )
        if len(weight) > 0:
            print ('loading weights for latent2albedo feature extraction network')
            latent2albedo.load_state_dict(torch.load(weight))
        return latent2albedo
    def build_latent2lit(self, weight = ''):
        latent2lit= th.nn.Sequential(
            LinearWN( 256 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.lit_dim )
        )
        if len(weight) > 0:
            print ('loading weights for latent2lit feature extraction network')
            latent2lit.load_state_dict(torch.load(weight))
        return latent2lit

    def _setup_renderer(self):
        mesh_file = '/home/uss00022/lelechen/basic/flame_data/data/head_template_mesh.obj'
        self.render = Renderer(self.image_size, obj_filename=mesh_file).to('cuda')
    
    def forward(self, latent, cam, pose, flameshape = None, flameexp= None, flametex= None, flamelit= None ):
        
        fea = self.Latent2fea(latent, axis = 1))
        shapecode = self.latent2shape(fea)
        expcode = self.latent2exp(fea)
        
        albedocode = self.latent2albedo(fea)
        litcode = self.latent2lit(fea)
        
        return_list = {}
        if self.opt.supervision =='render' or flameshape != None:
            vertices, landmarks2d, landmarks3d = self.flame(shape_params=shapecode, expression_params=expcode, pose_params=pose)
            trans_vertices = util.batch_orth_proj(vertices, cam)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]

            ## render
            albedos = self.flametex(albedocode, self.image_size) / 255.
            ops = self.render(vertices, trans_vertices, albedos, litcode.view(shape_latent.shape[0], 9,3))
            predicted_images = ops['images']

            return_list['landmarks3d'] = landmarks3d
            return_list['predicted_images'] = predicted_images
        else:
            return_list['expcode'] = expcode
            return_list['shapecode'] = shapecode
            return_list['litcode'] = litcode
            return_list['albedocode'] = albedocode
            
        if flameshape != None:
            flamelit = flamelit.view(-1, 9,3)        
            recons_vertices, _, recons_landmarks3d = self.flame(shape_params=flameshape, expression_params=flameexp, pose_params=pose)
            recons_trans_vertices = util.batch_orth_proj(recons_vertices, cam)
            recons_trans_vertices[..., 1:] = - recons_trans_vertices[..., 1:]

            ## render
            recons_albedos = self.flametex(flametex, self.image_size) / 255.
            recons_ops = self.render(recons_vertices, recons_trans_vertices, recons_albedos, flamelit)
            recons_images = recons_ops['images']
            return_list['recons_images'] = recons_images
            
        
        return return_list
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class RigNerft2(nn.Module):
    def __init__(self, flame_config, opt ):
        super().__init__()
        self.opt = opt
        
        self.nerf_latent_dim = 256
        self.gan_latent_dim = 512
        self.shape_dim = 100
        self.exp_dim = 50
        self.albedo_dim = 50
        self.lit_dim = 27

        self.flame_config = flame_config
        self.image_size = self.flame_config.image_size
        
        # funtion F networks
        latent2code = Latent2Code(flame_config, opt)

        self.Latent2ShapeExpCode, self.latent2shape, \
        self.latent2exp, self.Latent2AlbedoLitCode, \
        self.latent2albedo, self.latent2lit = self.get_f(latent2code)
        
        # rigNet
        # appearance part
        self.WGanEncoder = self.build_WGanEncoder(weight = '' if opt.isTrain else opt.WGanEncoder_weight )
        self.ShapeEncoder = self.build_ShapeEncoder(weight = '' if opt.isTrain else opt.ShapeEncoder_weight )
        self.ExpEncoder = self.build_ExpEncoder(weight = '' if opt.isTrain else opt.ExpEncoder_weight )
        self.WGanDecoder = self.build_WGanDecoder(weight = '' if opt.isTrain else opt.WGanDecoder_weight )
        # shape part
        self.WNerfEncoder = self.build_WNerfEncoder(weight = '' if opt.isTrain else opt.WNerfEncoder_weight )
        self.AlbedoEncoder = self.build_AlbedoEncoder(weight = '' if opt.isTrain else opt.AlbedoEncoder_weight )
        self.LitEncoder = self.build_LitEncoder(weight = '' if opt.isTrain else opt.LitEncoder_weight )
        self.WNerfDecoder = self.build_WNerfDecoder(weight = '' if opt.isTrain else opt.WNerfDecoder_weight )

        # Flame
        self.flame = FLAME(self.flame_config).to('cuda')
        self.flametex = FLAMETex(self.flame_config).to('cuda')
        self._setup_renderer()

        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.ckpt_path, exist_ok = True)

    def get_f(self,network):
        print (network)
        print ('loading weights for Latent2ShapeExpCode feature extraction network')
        network.Latent2ShapeExpCode.load_state_dict(torch.load(self.opt.Latent2ShapeExpCode_weight))
        print ('loading weights for latent2shape feature extraction network')
        network.latent2shape.load_state_dict(torch.load(self.opt.latent2shape_weight))
        print ('loading weights for latent2exp feature extraction network')
        network.latent2exp.load_state_dict(torch.load(self.opt.latent2exp_weight))
        print ('loading weights for Latent2AlbedoLitCode feature extraction network')
        network.Latent2AlbedoLitCode.load_state_dict(torch.load(self.opt.Latent2AlbedoLitCode_weight))
        print ('loading weights for latent2albedo feature extraction network')
        network.latent2albedo.load_state_dict(torch.load(self.opt.latent2albedo_weight))
        print ('loading weights for latent2albedo feature extraction network')
        network.latent2lit.load_state_dict(torch.load(self.opt.latent2lit_weight))
        
        return network.Latent2ShapeExpCode, network.latent2shape, network.latent2exp, \
               network.Latent2AlbedoLitCode, network.latent2albedo, network.latent2lit
    
    def latent2params(self, shape_latent, appearance_latent):
        shape_fea = self.Latent2ShapeExpCode(shape_latent)
        shapecode = self.latent2shape(shape_fea)
        expcode = self.latent2exp(shape_fea)
        
        app_fea = self.Latent2AlbedoLitCode(appearance_latent)
        albedocode = self.latent2albedo(app_fea)
        litcode = self.latent2lit(app_fea).view(shape_latent.shape[0], 9,3)
        paramset = [shapecode, expcode, albedocode, litcode]
        return paramset
    
    def build_WGanEncoder(self, weight = ''):
        WGanEncoder = th.nn.Sequential(
            LinearWN( self.gan_latent_dim , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for WGanEncoder  network')
            WGanEncoder.load_state_dict(torch.load(weight))
        return WGanEncoder
    
    def build_ShapeEncoder(self, weight = ''):
        ShapeEncoder = th.nn.Sequential(
            LinearWN( self.shape_dim , 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 128, 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for ShapeEncoder  network')
            ShapeEncoder.load_state_dict(torch.load(weight))
        return ShapeEncoder
    def build_ExpEncoder(self, weight = ''):
        ExpEncoder = th.nn.Sequential(
            LinearWN( self.exp_dim , 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 128, 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for ExpEncoder  network')
            ExpEncoder.load_state_dict(torch.load(weight))
        return ExpEncoder
    def build_WGanDecoder(self, weight = ''):
        WGanDecoder = th.nn.Sequential(
            LinearWN( 512 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.gan_latent_dim ),
        )
        if len(weight) > 0:
            print ('loading weights for WGanDecoder  network')
            WGanDecoder.load_state_dict(torch.load(weight))
        return WGanDecoder
    
    def build_WNerfEncoder(self, weight = ''):
        WNerfEncoder = th.nn.Sequential(
            LinearWN( self.nerf_latent_dim , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for WNerfEncoder  network')
            WNerfEncoder.load_state_dict(torch.load(weight))
        return WNerfEncoder
    
    def build_AlbedoEncoder(self, weight = ''):
        AlbedoEncoder = th.nn.Sequential(
            LinearWN( self.albedo_dim , 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 128, 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for AlbedoEncoder  network')
            AlbedoEncoder.load_state_dict(torch.load(weight))
        return AlbedoEncoder
    def build_LitEncoder(self, weight = ''):
        LitEncoder = th.nn.Sequential(
            LinearWN( self.lit_dim , 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 128, 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for LitEncoder  network')
            LitEncoder.load_state_dict(torch.load(weight))
        return LitEncoder
    def build_WNerfDecoder(self, weight = ''):
        WNerfDecoder = th.nn.Sequential(
            LinearWN( 512 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.nerf_latent_dim ),
        )
        if len(weight) > 0:
            print ('loading weights for WNerfDecoder  network')
            WNerfDecoder.load_state_dict(torch.load(weight))
        return WNerfDecoder

    def _setup_renderer(self):
        mesh_file = '/home/uss00022/lelechen/basic/flame_data/data/head_template_mesh.obj'
        self.render = Renderer(self.image_size, obj_filename=mesh_file).to('cuda')
    
    def rig(self,wgan, wnerf, p):
        shapecode, expcode, albedocode, litcode = p[0], p[1],p[2], p[3]
        lgan = self.WGanEncoder(wgan)
        lshape = self.ShapeEncoder(shapecode)
        lexp = self.ExpEncoder(expcode)
        deltagan = self.WGanDecoder(torch.cat([lgan, lshape, lexp], axis = 1))
        lnerf = self.WNerfEncoder(wnerf)
        lalbedo = self.AlbedoEncoder(albedocode)
        llit = self.LitEncoder(litcode.view(-1, 27))
        deltanerf = self.WNerfDecoder(torch.cat([lnerf, lalbedo, llit], axis = 1))

        return deltanerf + wnerf, deltagan + wgan
    
    def flame_render(self,p, pose, cam):
        shapecode,expcode,albedocode, litcode = p[0],p[1],p[2],p[3]
        vertices, landmarks2d, landmarks3d = self.flame(shape_params=shapecode, expression_params=expcode, pose_params=pose)
        trans_vertices = util.batch_orth_proj(vertices, cam)
        trans_vertices[..., 1:] = - trans_vertices[..., 1:]
        ## render
        albedos = self.flametex(albedocode, self.image_size) / 255.
        ops = self.render(vertices, trans_vertices, albedos, litcode)
        predicted_images = ops['images']
        
        return landmarks3d, predicted_images
    
    def forward(self, shape_latent_v, appearance_latent_v, shape_latent_w, appearance_latent_w, \
                    cam_v=None, pose_v=None, flameshape_v = None, flameexp_v = None, flametex_v = None,\
                    flamelit_v = None, cam_w=None, pose_w=None, flameshape_w = None, flameexp_w = None, flametex_w = None, flamelit_w = None):
        
        p_v = self.latent2params(shape_latent_v, appearance_latent_v)
        p_w = self.latent2params(shape_latent_w, appearance_latent_w)

        # if we input paired WGan and WNerf with P, output same WGan & Wnerf
        shape_latent_w_same, appearance_latent_w_same = self.rig(appearance_latent_w, shape_latent_w, p_w)
        p_w_same = self.latent2params(shape_latent_w_same, appearance_latent_w_same)

        # randomly choose one params to be edited
        choice = torch.randint(0, 4 ,(1,)).item()
        
        # if we input WGan and Wnerf, and P_v, output hat_WGan, hat_WNerf
        p_w_replaced = []
        for i in range(4):
            if i != choice:
                p_w_replaced.append(p_w[i])
            else:
                p_w_replaced.append(p_v[i])

        shape_latent_w_hat, appearance_latent_w_hat = self.rig(appearance_latent_w, shape_latent_w, p_w_replaced)
        # map chagned w back to P
        p_w_mapped = self.latent2params(shape_latent_w_hat, appearance_latent_w_hat)

        p_v_ = []
        p_w_ = []
        for j in range(4):
            if j != choice:
                p_w_.append(p_w_mapped[j])
                p_v_.append(p_v[j])
            else:
                p_w_.append(p_w[j])
                p_v_.append(p_w_mapped[j])
        
        landmark_same, render_img_same = self.flame_render(p_w_same, pose_w, cam_w)
        landmark_w_, render_img_w_ = self.flame_render(p_w_, pose_w, cam_w)
        landmark_v_, render_img_v_ = self.flame_render(p_v_, pose_v, cam_v)

        if flameshape_v != None:
            p_v_vis = [flameshape_v, flameexp_v, flametex_v, flamelit_v.view(-1, 9,3)] 
            p_w_vis = [flameshape_w, flameexp_w, flametex_w, flamelit_w.view(-1, 9,3)] 
            _, recons_images_v = self.flame_render(p_v_vis, pose_v, cam_v)
            _, recons_images_w = self.flame_render(p_w_vis, pose_w, cam_w)

        else:
            recons_images_v = render_img_w_
            recons_images_w = render_img_w_

        return landmark_same, render_img_same, \
                landmark_w_, render_img_w_ , \
                landmark_v_, render_img_v_ , \
                recons_images_v, recons_images_w 
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



class RigNerft(nn.Module):
    def __init__(self, flame_config, opt ):
        super().__init__()
        self.opt = opt
        
        self.nerf_latent_dim = 256
        self.gan_latent_dim = 512
        self.shape_dim = 100
        self.exp_dim = 50
        self.albedo_dim = 50
        self.lit_dim = 27

        self.flame_config = flame_config
        self.image_size = self.flame_config.image_size
        
        # funtion F networks
        latent2code = Latent2Code(flame_config, opt)

        self.Latent2ShapeExpCode, self.latent2shape, \
        self.latent2exp, self.Latent2AlbedoLitCode, \
        self.latent2albedo, self.latent2lit = self.get_f(latent2code)
        
        # rigNet
        # appearance part
        self.WGanEncoder = self.build_WGanEncoder(weight = '' if opt.isTrain else opt.WGanEncoder_weight )
        self.ShapeEncoder = self.build_ShapeEncoder(weight = '' if opt.isTrain else opt.ShapeEncoder_weight )
        self.ExpEncoder = self.build_ExpEncoder(weight = '' if opt.isTrain else opt.ExpEncoder_weight )
        self.WGanDecoder = self.build_WGanDecoder(weight = '' if opt.isTrain else opt.WGanDecoder_weight )
        # shape part
        # self.WNerfEncoder = self.build_WNerfEncoder(weight = '' if opt.isTrain else opt.WNerfEncoder_weight )
        self.AlbedoEncoder = self.build_AlbedoEncoder(weight = '' if opt.isTrain else opt.AlbedoEncoder_weight )
        self.LitEncoder = self.build_LitEncoder(weight = '' if opt.isTrain else opt.LitEncoder_weight )
        self.WNerfDecoder = self.build_WNerfDecoder(weight = '' if opt.isTrain else opt.WNerfDecoder_weight )

        # Flame
        self.flame = FLAME(self.flame_config).to('cuda')
        self.flametex = FLAMETex(self.flame_config).to('cuda')
        self._setup_renderer()

        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.ckpt_path, exist_ok = True)

    def get_f(self,network):
        print (network)
        print ('loading weights for Latent2ShapeExpCode feature extraction network')
        network.Latent2ShapeExpCode.load_state_dict(torch.load(self.opt.Latent2ShapeExpCode_weight))
        print ('loading weights for latent2shape feature extraction network')
        network.latent2shape.load_state_dict(torch.load(self.opt.latent2shape_weight))
        print ('loading weights for latent2exp feature extraction network')
        network.latent2exp.load_state_dict(torch.load(self.opt.latent2exp_weight))
        print ('loading weights for Latent2AlbedoLitCode feature extraction network')
        network.Latent2AlbedoLitCode.load_state_dict(torch.load(self.opt.Latent2AlbedoLitCode_weight))
        print ('loading weights for latent2albedo feature extraction network')
        network.latent2albedo.load_state_dict(torch.load(self.opt.latent2albedo_weight))
        print ('loading weights for latent2albedo feature extraction network')
        network.latent2lit.load_state_dict(torch.load(self.opt.latent2lit_weight))
        
        return network.Latent2ShapeExpCode, network.latent2shape, network.latent2exp, \
               network.Latent2AlbedoLitCode, network.latent2albedo, network.latent2lit
    
    def latent2params(self, shape_latent, appearance_latent):
        shape_fea = self.Latent2ShapeExpCode(shape_latent)
        shapecode = self.latent2shape(shape_fea)
        expcode = self.latent2exp(shape_fea)
        
        app_fea = self.Latent2AlbedoLitCode(appearance_latent)
        albedocode = self.latent2albedo(app_fea)
        litcode = self.latent2lit(app_fea).view(shape_latent.shape[0], 9,3)
        paramset = [shapecode, expcode, albedocode, litcode]
        return paramset
    
    def build_WGanEncoder(self, weight = ''):
        WGanEncoder = th.nn.Sequential(
            LinearWN( self.gan_latent_dim , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for WGanEncoder  network')
            WGanEncoder.load_state_dict(torch.load(weight))
        return WGanEncoder
    
    def build_ShapeEncoder(self, weight = ''):
        ShapeEncoder = th.nn.Sequential(
            LinearWN( self.shape_dim , 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 128, 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for ShapeEncoder  network')
            ShapeEncoder.load_state_dict(torch.load(weight))
        return ShapeEncoder
    def build_ExpEncoder(self, weight = ''):
        ExpEncoder = th.nn.Sequential(
            LinearWN( self.exp_dim , 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 128, 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for ExpEncoder  network')
            ExpEncoder.load_state_dict(torch.load(weight))
        return ExpEncoder
    def build_WGanDecoder(self, weight = ''):
        WGanDecoder = th.nn.Sequential(
            LinearWN( 512 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.gan_latent_dim ),
        )
        if len(weight) > 0:
            print ('loading weights for WGanDecoder  network')
            WGanDecoder.load_state_dict(torch.load(weight))
        return WGanDecoder
    
    def build_WNerfEncoder(self, weight = ''):
        WNerfEncoder = th.nn.Sequential(
            LinearWN( self.nerf_latent_dim , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for WNerfEncoder  network')
            WNerfEncoder.load_state_dict(torch.load(weight))
        return WNerfEncoder
    
    def build_AlbedoEncoder(self, weight = ''):
        AlbedoEncoder = th.nn.Sequential(
            LinearWN( self.albedo_dim , 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 128, 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for AlbedoEncoder  network')
            AlbedoEncoder.load_state_dict(torch.load(weight))
        return AlbedoEncoder
    def build_LitEncoder(self, weight = ''):
        LitEncoder = th.nn.Sequential(
            LinearWN( self.lit_dim , 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 128, 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for LitEncoder  network')
            LitEncoder.load_state_dict(torch.load(weight))
        return LitEncoder
    def build_WNerfDecoder(self, weight = ''):
        WNerfDecoder = th.nn.Sequential(
            LinearWN( 512 , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.nerf_latent_dim ),
        )
        if len(weight) > 0:
            print ('loading weights for WNerfDecoder  network')
            WNerfDecoder.load_state_dict(torch.load(weight))
        return WNerfDecoder

    def _setup_renderer(self):
        mesh_file = '/home/uss00022/lelechen/basic/flame_data/data/head_template_mesh.obj'
        self.render = Renderer(self.image_size, obj_filename=mesh_file).to('cuda')
    
    def rig(self,wgan, wnerf, p):
        shapecode, expcode, albedocode, litcode = p[0], p[1],p[2], p[3]
        lgan = self.WGanEncoder(wgan)
        lshape = self.ShapeEncoder(shapecode)
        lexp = self.ExpEncoder(expcode)
        deltagan = self.WGanDecoder(torch.cat([lgan, lshape, lexp], axis = 1))
        lnerf = self.WNerfEncoder(wnerf)
        lalbedo = self.AlbedoEncoder(albedocode)
        llit = self.LitEncoder(litcode.view(-1, 27))
        deltanerf = self.WNerfDecoder(torch.cat([lnerf, lalbedo, llit], axis = 1))

        return deltanerf + wnerf, deltagan + wgan
    
    def flame_render(self,p, pose, cam):
        shapecode,expcode,albedocode, litcode = p[0],p[1],p[2],p[3]
        vertices, landmarks2d, landmarks3d = self.flame(shape_params=shapecode, expression_params=expcode, pose_params=pose)
        trans_vertices = util.batch_orth_proj(vertices, cam)
        trans_vertices[..., 1:] = - trans_vertices[..., 1:]
        ## render
        albedos = self.flametex(albedocode, self.image_size) / 255.
        ops = self.render(vertices, trans_vertices, albedos, litcode)
        predicted_images = ops['images']
        
        return landmarks3d, predicted_images
    
    def forward(self, shape_latent_v, appearance_latent_v, shape_latent_w, appearance_latent_w, \
                    cam_v=None, pose_v=None, flameshape_v = None, flameexp_v = None, flametex_v = None,\
                    flamelit_v = None, cam_w=None, pose_w=None, flameshape_w = None, flameexp_w = None, flametex_w = None, flamelit_w = None):
        
        p_v = self.latent2params(shape_latent_v, appearance_latent_v)
        p_w = self.latent2params(shape_latent_w, appearance_latent_w)

        # if we input paired WGan and WNerf with P, output same WGan & Wnerf
        shape_latent_w_same, appearance_latent_w_same = self.rig(appearance_latent_w, shape_latent_w, p_w)
        p_w_same = self.latent2params(shape_latent_w_same, appearance_latent_w_same)

        # randomly choose one params to be edited
        choice = torch.randint(0, 4 ,(1,)).item()
        
        # if we input WGan and Wnerf, and P_v, output hat_WGan, hat_WNerf
        p_w_replaced = []
        for i in range(4):
            if i != choice:
                p_w_replaced.append(p_w[i])
            else:
                p_w_replaced.append(p_v[i])

        shape_latent_w_hat, appearance_latent_w_hat = self.rig(appearance_latent_w, shape_latent_w, p_w_replaced)
        # map chagned w back to P
        p_w_mapped = self.latent2params(shape_latent_w_hat, appearance_latent_w_hat)

        p_v_ = []
        p_w_ = []
        for j in range(4):
            if j != choice:
                p_w_.append(p_w_mapped[j])
                p_v_.append(p_v[j])
            else:
                p_w_.append(p_w[j])
                p_v_.append(p_w_mapped[j])
        
        landmark_same, render_img_same = self.flame_render(p_w_same, pose_w, cam_w)
        landmark_w_, render_img_w_ = self.flame_render(p_w_, pose_w, cam_w)
        landmark_v_, render_img_v_ = self.flame_render(p_v_, pose_v, cam_v)

        if flameshape_v != None:
            p_v_vis = [flameshape_v, flameexp_v, flametex_v, flamelit_v.view(-1, 9,3)] 
            p_w_vis = [flameshape_w, flameexp_w, flametex_w, flamelit_w.view(-1, 9,3)] 
            _, recons_images_v = self.flame_render(p_v_vis, pose_v, cam_v)
            _, recons_images_w = self.flame_render(p_w_vis, pose_w, cam_w)

        else:
            recons_images_v = render_img_w_
            recons_images_w = render_img_w_

        return landmark_same, render_img_same, \
                landmark_w_, render_img_w_ , \
                landmark_v_, render_img_v_ , \
                recons_images_v, recons_images_w 
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
