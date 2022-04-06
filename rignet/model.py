import numpy as np
import torch.nn.functional as F
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
sys.path.append('/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/')
from Flamerenderer import FlameRenderer
import util
from models.FLAME import FLAME, FLAMETex
sys.path.append('/home/uss00022/lelechen/github/CIPS-3D/utils')
from visualizer import Visualizer
import tensor_util
from blocks import *
import face_alignment

sys.path.append('/home/uss00022/lelechen/github/StyleNeRF')
import dnnlib
import training
import torch_utils
import legacy

class Latent2Code(nn.Module):
    def __init__(self, flame_config, opt ):
        super().__init__()
        self.opt = opt
        # self.save_hyperparameters()
        self.flame_config = flame_config
    
        self.image_size = self.flame_config.image_size
        # networks
        self.latent_dim = 512 * 21
        self.shape_dim = 100
        self.exp_dim = 50
        self.albedo_dim = 50
        self.lit_dim = 27
        self.pose_dim = 6
        self.Latent2fea = self.build_Latent2CodeFea( weight = '' if opt.isTrain else opt.Latent2ShapeExpCode_weight)
        self.latent2shape = self.build_latent2shape( weight = '' if opt.isTrain else opt.latent2shape_weight)
        self.latent2exp = self.build_latent2exp(weight = '' if opt.isTrain else opt.latent2exp_weight)
        self.latent2albedo = self.build_latent2albedo(weight = '' if opt.isTrain else opt.latent2albedo_weight)
        self.latent2lit = self.build_latent2lit(weight = '' if opt.isTrain else opt.latent2lit_weight)
        # self.latent2pose = self.build_latent2pose(weight = '' if opt.isTrain else opt.latent2poses_weight)

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

    # def build_latent2pose(self, weight = ''):
    #     latent2pose= th.nn.Sequential(
    #         LinearWN( 256 , 256 ),
    #         th.nn.LeakyReLU( 0.2, inplace = True ),
    #         LinearWN( 256, self.pose_dim )
    #     )
    #     if len(weight) > 0:
    #         print ('loading weights for latent2pose feature extraction network')
    #         latent2pose.load_state_dict(torch.load(weight))
    #     return latent2pose
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
        self.render = FlameRenderer(self.image_size, obj_filename=mesh_file).to('cuda')
    
    def forward(self, latent, cam, pose, flameshape = None, flameexp= None, flametex= None, flamelit= None ):
        fea = self.Latent2fea(latent)
        shapecode = self.latent2shape(fea)
        expcode = self.latent2exp(fea)
        
        albedocode = self.latent2albedo(fea)
        litcode = self.latent2lit(fea)
        # posecode = self.latent2pose(fea)
        return_list = {}
        return_list['expcode'] = expcode
        return_list['shapecode'] = shapecode
        return_list['litcode'] = litcode
        return_list['albedocode'] = albedocode

        if self.opt.supervision =='render' or flameshape != None:
            vertices, landmarks2d, landmarks3d = self.flame(shape_params=shapecode, expression_params=expcode, pose_params=pose)
            trans_vertices = util.batch_orth_proj(vertices, cam)
            trans_vertices[..., 1:] = - trans_vertices[..., 1:]

            ## render
            albedos = self.flametex(albedocode, self.image_size) / 255.
            ops = self.render(vertices, trans_vertices, albedos, litcode.view(-1, 9,3))
            predicted_images = ops['images']

            return_list['landmarks3d'] = landmarks3d
            return_list['predicted_images'] = predicted_images
                        
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

class RigNerft(nn.Module):
    def __init__(self, flame_config, opt ):
        super().__init__()
        self.opt = opt
        
        self.latent_dim = 512 * 21
        self.shape_dim = 100
        self.exp_dim = 50
        self.albedo_dim = 50
        self.lit_dim = 27

        self.latent_fea_dim = 256
        self.param_fea_dim = 256

        self.flame_config = flame_config
        self.image_size = self.flame_config.image_size
        
        # funtion F networks
        latent2code = Latent2Code(flame_config, opt)

        self.Latent2fea, self.latent2shape, \
        self.latent2exp, self.latent2albedo, self.latent2lit = self.get_f(latent2code)
        
        # rigNet
        self.LatentEncoder = self.build_WEncoder(weight = '' if opt.isTrain else opt.WEncoder_weight )
        self.ParamEncoder = self.build_ParamEncoder(weight = '' if opt.isTrain else opt.ParamEncoder_weight )
        self.LatentDecoder = self.build_WDecoder(weight = '' if opt.isTrain else opt.WDecoder_weight )
       
        # Flame
        self.flame = FLAME(self.flame_config).to('cuda')
        self.flametex = FLAMETex(self.flame_config).to('cuda')
        self._setup_renderer()

        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.ckpt_path, exist_ok = True)

        if not self.opt.isTrain:
            with dnnlib.util.open_url( self.opt.nerfpkl) as f:
                network = legacy.load_network_pkl(f)
                G = network['G_ema'].to('cuda') # type: ignore
            
            # avoid persistent classes... 
            from training.networks import Generator
            # from training.stylenerf import Discriminator
            from torch_utils import misc
            with torch.no_grad():
                self.G2 = Generator(*G.init_args, **G.init_kwargs).to('cuda')
                misc.copy_params_and_buffers(G, self.G2, require_all=False)


    def get_f(self,network):
        print (network)
        print ('loading weights for Latent2fea feature extraction network, ' + self.opt.Latent2ShapeExpCode_weight)
        network.Latent2fea.load_state_dict(torch.load(self.opt.Latent2ShapeExpCode_weight))
        print ('loading weights for latent2shape feature extraction network, ' + self.opt.latent2shape_weight)
        network.latent2shape.load_state_dict(torch.load(self.opt.latent2shape_weight))
        print ('loading weights for latent2exp feature extraction network, ' + self.opt.latent2exp_weight)
        network.latent2exp.load_state_dict(torch.load(self.opt.latent2exp_weight))
        print ('loading weights for latent2albedo feature extraction network, ' + self.opt.latent2albedo_weight)
        network.latent2albedo.load_state_dict(torch.load(self.opt.latent2albedo_weight))
        print ('loading weights for latent2albedo feature extraction network, ' + self.opt.latent2lit_weight)
        network.latent2lit.load_state_dict(torch.load(self.opt.latent2lit_weight))
        
        return network.Latent2fea, network.latent2shape, network.latent2exp, network.latent2albedo, network.latent2lit
    
    def latent2params(self, latent):
        fea = self.Latent2fea(latent)

        shapecode = self.latent2shape(fea)
        expcode = self.latent2exp(fea)
        albedocode = self.latent2albedo(fea)
        litcode = self.latent2lit(fea).view(-1, 9,3)
        
        paramset = [shapecode, expcode, albedocode, litcode]
        return paramset
    
    def build_WEncoder(self, weight = ''):
        WEncoder = th.nn.Sequential(
            LinearWN( self.latent_dim , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.latent_fea_dim ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for WEncoder  network, ' +weight )
            WEncoder.load_state_dict(torch.load(weight))
        return WEncoder
    
    def build_ParamEncoder(self, weight = ''):
        ParamEncoder = th.nn.Sequential(
            LinearWN( self.shape_dim + self.exp_dim + self.lit_dim + self.albedo_dim, 128 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 128, self.param_fea_dim ),
            th.nn.LeakyReLU( 0.2, inplace = True )
        )
        if len(weight) > 0:
            print ('loading weights for ParamEncoder  network, ' +weight )
            ParamEncoder.load_state_dict(torch.load(weight))
        return ParamEncoder

    def build_WDecoder(self, weight = ''):
        WDecoder = th.nn.Sequential(
            LinearWN( self.latent_fea_dim + self.param_fea_dim , 256 ),
            th.nn.LeakyReLU( 0.2, inplace = True ),
            LinearWN( 256, self.latent_dim ),
        )
        if len(weight) > 0:
            print ('loading weights for WDecoder  network, ' +weight )
            WDecoder.load_state_dict(torch.load(weight))
        return WDecoder
    
  
    def _setup_renderer(self):
        mesh_file = '/home/uss00022/lelechen/basic/flame_data/data/head_template_mesh.obj'
        self.render = FlameRenderer(self.image_size, obj_filename=mesh_file).to('cuda')
    
    def rig(self,w, p):
        shapecode, expcode, albedocode, litcode = p[0], p[1],p[2], p[3].view(-1, 27)

        l_w = self.LatentEncoder(w)
       
        l_p = self.ParamEncoder(torch.cat([shapecode, expcode, albedocode, litcode], axis = 1))
       
        delta_w = self.LatentDecoder(torch.cat([l_p, l_w], axis = 1))

        return  delta_w + w
    
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
    
    def forward(self, latent_v, latent_w, \
                    cam_v=None, pose_v=None, flameshape_v = None, flameexp_v = None, flametex_v = None,\
                    flamelit_v = None, cam_w=None, pose_w=None, flameshape_w = None, flameexp_w = None, flametex_w = None, flamelit_w = None):
        
        p_v = self.latent2params(latent_v)
        p_w = self.latent2params(latent_w)

        # if we input paired W with P, output same W
        latent_w_same = self.rig(latent_w,  p_w)
        p_w_same = self.latent2params(latent_w_same)

        # randomly choose one params to be edited
        choice = torch.randint(0, 4 ,(1,)).item()
        
        # if we input W, and P_v, output hat_W
        p_w_replaced = []
        for i in range(4):
            if i != choice:
                p_w_replaced.append(p_w[i])
            else:
                p_w_replaced.append(p_v[i])

        latent_w_hat = self.rig(latent_w, p_w_replaced)
        # map chagned w back to P
        p_w_mapped = self.latent2params(latent_w_hat)

        p_v_ = []
        p_w_ = []
        for j in range(4):
            if j != choice:
                p_w_.append(p_w_mapped[j])
                p_v_.append(p_v[j])
            else:
                p_w_.append(p_w[j])
                p_v_.append(p_w_mapped[j])
        
        landmark_w_, render_img_w_ = self.flame_render(p_w_, pose_w, cam_w)
        landmark_v_, render_img_v_ = self.flame_render(p_v_, pose_v, cam_v)
        return_list = {}


        if flameshape_v != None:
            landmark_same, render_img_same = self.flame_render(p_w_same, pose_w, cam_w)
            p_v_vis = [flameshape_v, flameexp_v, flametex_v, flamelit_v.view(-1, 9,3)] 
            p_w_vis = [flameshape_w, flameexp_w, flametex_w, flamelit_w.view(-1, 9,3)] 
            _, recons_images_v = self.flame_render(p_v_vis, pose_v, cam_v)
            _, recons_images_w = self.flame_render(p_w_vis, pose_w, cam_w)

            return_list['recons_images_v'] = recons_images_v
            return_list['recons_images_w'] = recons_images_w
            return_list['landmark_same'] = landmark_same
            return_list['render_img_same'] = render_img_same

        return_list['latent_w_same'] = latent_w_same
        return_list['landmark_w_'] = landmark_w_
        return_list['render_img_w_'] = render_img_w_
        return_list['landmark_v_'] = landmark_v_
        return_list['render_img_v_'] = render_img_v_
        

        return return_list
    
    def test(self, latent_v, latent_w, \
                    cam_v=None, pose_v=None, flameshape_v = None, flameexp_v = None, flametex_v = None,\
                    flamelit_v = None, cam_w=None, pose_w=None, flameshape_w = None, flameexp_w = None, flametex_w = None, flamelit_w = None):
        
        syns_v = self.G2.forward(styles = latent_v.view(-1, 21,512))['img']

        syns_w = self.G2.forward(styles = latent_w.view(-1, 21,512))['img']

        p_v = self.latent2params(latent_v)
    
        p_w = self.latent2params(latent_w)

        # debug:

        # if we input paired W with P, output same W
        latent_w_same = self.rig(latent_w,  p_w)

        syns_w_same = self.G2.forward(styles = latent_w_same.view(-1, 21,512))['img']

        p_w_same = self.latent2params(latent_w_same)

        # randomly choose one params to be edited
        choice = torch.randint(0, 4 ,(1,)).item()
        
        # if we input W, and P_v, output hat_W
        p_w_replaced = []
        for i in range(4):
            if i != choice:
                p_w_replaced.append(p_w[i])
            else:
                p_w_replaced.append(p_v[i])

        latent_w_hat = self.rig(latent_w, p_w_replaced)

        syns_w_hat = self.G2.forward(styles = latent_w_hat.view(-1, 21,512))['img']

        # map chagned w back to P
        p_w_mapped = self.latent2params(latent_w_hat)

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

        p_v_vis = [flameshape_v, flameexp_v, flametex_v, flamelit_v.view(-1, 9,3)] 
        p_w_vis = [flameshape_w, flameexp_w, flametex_w, flamelit_w.view(-1, 9,3)] 
        _, recons_images_v = self.flame_render(p_v_vis, pose_v, cam_v)
        _, recons_images_w = self.flame_render(p_w_vis, pose_w, cam_w)

        _, recons_images_hat = self.flame_render(p_w_mapped, pose_w, cam_w)


        return_list = {}
        return_list['landmark_same'] = landmark_same
        return_list['render_img_same'] = render_img_same
        return_list['landmark_w_'] = landmark_w_
        return_list['render_img_w_'] = render_img_w_

        return_list['landmark_v_'] = landmark_v_
        return_list['render_img_v_'] = render_img_v_

        return_list['recons_images_v'] = recons_images_v
        return_list['recons_images_w'] = recons_images_w
        return_list['choice'] = choice
        return_list['syns_v'] = syns_v
        return_list['syns_w'] = syns_w
        return_list['syns_w_same'] = syns_w_same

        return_list['syns_w_hat'] = syns_w_hat
        return_list['recons_images_hat'] = recons_images_hat
      

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
