
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from model import *
sys.path.append('/home/uss00022/lelechen/github/CIPS-3D/utils')
from visualizer import Visualizer
import util
from dataset import *
import time 

class RigModule():
    def __init__(self, flame_config, opt ):
        super().__init__()
        self.opt = opt
        self.flame_config = flame_config
        self.visualizer = Visualizer(opt)
        if opt.cuda:
            self.device = torch.device("cuda")
        self.rig = RigNerft( flame_config, opt)
        print (self.rig)
        self.optimizer = optim.Adam( list(self.rig.WGanEncoder.parameters()) + \
                                  list(self.rig.ShapeEncoder.parameters()) + \
                                  list(self.rig.ExpEncoder.parameters()) + \
                                  list(self.rig.WGanDecoder.parameters()) + \
                                  list(self.rig.WNerfEncoder.parameters()) + \
                                  list(self.rig.AlbedoEncoder.parameters()) + \
                                  list(self.rig.LitEncoder.parameters()) + \
                                  list(self.rig.WNerfDecoder.parameters()) \
                                  , lr= self.opt.lr , betas=(self.opt.beta1, 0.999))
        for p in self.rig.Latent2ShapeExpCode.parameters():
            p.requires_grad = False 
        for p in self.rig.Latent2AlbedoLitCode.parameters():
            p.requires_grad = False 
        for p in self.rig.latent2shape.parameters():
            p.requires_grad = False 
        for p in self.rig.latent2exp.parameters():
            p.requires_grad = False 
        for p in self.rig.latent2albedo.parameters():
            p.requires_grad = False 
        for p in self.rig.latent2lit.parameters():
            p.requires_grad = False 
        for p in self.rig.flame.parameters():
            p.requires_grad = False    
       


        if opt.isTrain:
            self.rig =torch.nn.DataParallel(self.rig, device_ids=range(len(self.opt.gpu_ids)))
        self.rig = self.rig.to(self.device)
        if opt.name == 'rig':
            self.dataset  = FFHQRigDataset(opt)
        else:
            print ('!!!!!!!!!!WRONG name for dataset')
        
        self.data_loader = DataLoaderWithPrefetch(self.dataset, \
                    batch_size=opt.batchSize,\
                    drop_last=opt.isTrain,\
                    shuffle = opt.isTrain,\
                    num_workers = opt.nThreads, \
                    prefetch_size = min(8, opt.nThreads))
      
        print ('========', len(self.data_loader),'========')
        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.ckpt_path, exist_ok = True)

    def train(self):
        t0 = time.time()
        for epoch in range( 1000):
            for step, batch in enumerate(tqdm(self.data_loader)):
                t1 = time.time()
                landmark_same, render_img_same, \
                landmark_w_, render_img_w_ , \
                landmark_v_, render_img_v_ , \
                recons_images_v, recons_images_w \
                = self.rig.forward(
                            batch[0]['shape_latent'].to(self.device),
                            batch[0]['appearance_latent'].to(self.device),
                            batch[1]['shape_latent'].to(self.device),
                            batch[1]['appearance_latent'].to(self.device),
                            
                            batch[0]['cam'].to(self.device), 
                            batch[0]['pose'].to(self.device),
                            batch[0]['shape'].to(self.device),
                            batch[0]['exp'].to(self.device),
                            batch[0]['tex'].to(self.device),
                            batch[0]['lit'].to(self.device),

                            batch[1]['cam'].to(self.device), 
                            batch[1]['pose'].to(self.device),
                            
                            batch[1]['shape'].to(self.device),
                            batch[1]['exp'].to(self.device),
                            batch[1]['tex'].to(self.device),
                            batch[1]['lit'].to(self.device)
                            )
                t2 = time.time()
                losses = {}
                # keep batch[1], w the same
                losses['landmark_same'] = util.l2_distance(landmark_same[:, 17:, :2], batch[1]['gt_landmark'][:, 17:, :2].to(self.device)) * self.flame_config.w_lmks
                losses['photometric_texture_same'] = (batch[1]['img_mask'].to(self.device) * (render_img_same - batch[1]['gt_image'].to(self.device) ).abs()).mean() * self.flame_config.w_pho
                
                # close to w
                losses['landmark_w_'] = util.l2_distance(landmark_w_[:, 17:, :2], batch[1]['gt_landmark'][:, 17:, :2].to(self.device)) * self.flame_config.w_lmks
                losses['photometric_texture_w_'] = (batch[1]['img_mask'].to(self.device) * (render_img_w_ - batch[1]['gt_image'].to(self.device) ).abs()).mean() * self.flame_config.w_pho
                
                # close to v
                losses['landmark_v_'] = util.l2_distance(landmark_v_[:, 17:, :2], batch[0]['gt_landmark'][:, 17:, :2].to(self.device)) * self.flame_config.w_lmks
                losses['photometric_texture_v_'] = (batch[0]['img_mask'].to(self.device) * (render_img_v_ - batch[0]['gt_image'].to(self.device) ).abs()).mean() * self.flame_config.w_pho
                
                loss = losses['landmark_same'] + losses['photometric_texture_same'] + \
                       losses['landmark_w_'] + losses['photometric_texture_w_'] + \
                       losses['landmark_v_'] + losses['photometric_texture_v_']
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tqdm_dict = {'landmark_same': losses['landmark_same'].data, \
                             'photometric_texture_same': losses['photometric_texture_same'].data, \
                             'landmark_w_': losses['landmark_w_'].data, \
                             'photometric_texture_w_': losses['photometric_texture_w_'].data, \
                             'landmark_v_': losses['landmark_v_'].data, \
                             'photometric_texture_v_': losses['photometric_texture_v_'].data
                               }
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in tqdm_dict.items()} 
                t3 = time.time()
                self.visualizer.print_current_errors(epoch, step, errors, t1-t0, t2-t1, t3-t2)
                t0 = time.time()
            if epoch % self.opt.save_step == 0:  
                
                visind = 0
                # visualize the image close to v
                image_v = vis_tensor(image_tensor= batch[0]['gt_image'], 
                                        image_path = batch[0]['image_path'][0] +'---V',
                                        device = self.device
                                         )

                lmark_v = vis_tensor(image_tensor= batch[0]['gt_image'], 
                                        image_path = batch[0]['image_path'][0] +'--V',
                                        land_tensor = batch[0]['gt_landmark'],
                                        cam = batch[0]['cam'], 
                                        device = self.device
                                         )
               

                image_w = vis_tensor(image_tensor= batch[1]['gt_image'], 
                                        image_path = batch[1]['image_path'][0] +'---W',
                                        device = self.device
                                         )

                lmark_w = vis_tensor(image_tensor= batch[1]['gt_image'], 
                                        image_path = batch[1]['image_path'][0] +'--W',
                                        land_tensor = batch[1]['gt_landmark'],
                                        cam = batch[1]['cam'], 
                                        device = self.device
                                         )

                recons_images_w = vis_tensor(image_tensor= recons_images_w, 
                                        image_path = batch[1]['image_path'][0] +'---recons-W',
                                        device = self.device
                                         )
                recons_images_v = vis_tensor(image_tensor= recons_images_v, 
                                        image_path = batch[0]['image_path'][0] +'---recons-V',
                                        device = self.device
                                         )

                genlmark_same = vis_tensor(image_tensor= batch[1]['gt_image'], 
                                        image_path = batch[1]['image_path'][0] +'---same-W',
                                        land_tensor = landmark_same,
                                        cam = batch[1]['cam'], 
                                        device = self.device
                                         )
        
                genimage_same = vis_tensor(image_tensor= render_img_same, 
                                        image_path = batch[1]['image_path'][0] +'---same-V',
                                        device = self.device
                                         )
          
                genlmark_w = vis_tensor(image_tensor= batch[1]['gt_image'], 
                                        image_path = batch[1]['image_path'][0] +'---close-W',
                                        land_tensor = landmark_w_,
                                        cam = batch[1]['cam'], 
                                        device = self.device
                                         )

                genimage_w = vis_tensor(image_tensor= render_img_w_, 
                                        image_path = batch[1]['image_path'][0] +'---close-W',
                                        device = self.device
                                         )

                genlmark_v = vis_tensor(image_tensor= batch[0]['gt_image'], 
                                        image_path = batch[0]['image_path'][0] +'---close-V',
                                        land_tensor = landmark_v_,
                                        cam = batch[0]['cam'], 
                                        device = self.device
                                         )
                genimage_v = vis_tensor(image_tensor = render_img_v_, 
                                        image_path = batch[0]['image_path'][0]+'---close-V', 
                                        device = self.device)

                
                visuals = OrderedDict([
                ('image_v', image_v),
                ('lmark_v', lmark_v),
                ('recons_images_v', recons_images_v),

                ('image_w', image_w),
                ('lmark_w', lmark_w),
                ('recons_images_w', recons_images_w),
                
                ('genlmark_same_W', genlmark_same ),
                ('genimage_same_W', genimage_same),

                ('genlmark_w', genlmark_w),
                ('genimage_w', genimage_w ),

                ('genlmark_v', genlmark_v),
                ('genimage_v', genimage_v )
                ])
        
                self.visualizer.display_current_results(visuals, epoch, self.opt.save_step) 

                torch.save(self.rig.module.WGanEncoder.state_dict(), self.opt.WGanEncoder_weight)
                torch.save(self.rig.module.ShapeEncoder.state_dict(),self.opt.ShapeEncoder_weight)
                torch.save(self.rig.module.ExpEncoder.state_dict(), self.opt.ExpEncoder_weight)
                torch.save(self.rig.module.WGanDecoder.state_dict(), self.opt.WGanDecoder_weight)
                
                torch.save(self.rig.module.WNerfEncoder.state_dict(), self.opt.WNerfEncoder_weight)
                torch.save(self.rig.module.AlbedoEncoder.state_dict(),self.opt.AlbedoEncoder_weight)
                torch.save(self.rig.module.LitEncoder.state_dict(),self.opt.LitEncoder_weight)
                torch.save(self.rig.module.WNerfDecoder.state_dict(),self.opt.WNerfDecoder_weight)
    def test(self):
        for p in self.latent2code.parameters():
            p.requires_grad = False 
        for step, batch in enumerate(tqdm(self.data_loader)):
            with torch.no_grad():    
                landmarks3d, predicted_images = self.latent2code.forward(
                        batch['shape_latent'].to(self.device), \
                        batch['appearance_latent'].to(self.device), \
                        batch['cam'].to(self.device), batch['pose'].to(self.device))
            losses = {}
            losses['landmark'] = util.l2_distance(landmarks3d[:, 17:, :2], batch['gt_landmark'][:, 17:, :2].to(self.device)) * self.flame_config.w_lmks
            losses['photometric_texture'] = (batch['img_mask'].to(self.device) * (predicted_images - batch['gt_image'].to(self.device) ).abs()).mean() * self.flame_config.w_pho
            loss = losses['landmark'] + losses['photometric_texture']
            
            tqdm_dict = {'loss_landmark': losses['landmark'].data, 'loss_tex': losses['photometric_texture'].data  }
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in tqdm_dict.items()} 
            self.visualizer.print_current_errors(0, step, errors, 0)

            visind = 0
            gtimage = batch['gt_image'].data[visind].cpu()
            gtimage = tensor_util.tensor2im(gtimage  , normalize = False)
            gtimage = np.ascontiguousarray(gtimage, dtype=np.uint8)
            gtimage = tensor_util.writeText(gtimage, batch['image_path'][visind])
            gtimage = np.ascontiguousarray(gtimage, dtype=np.uint8)
            gtimage = np.clip(gtimage, 0, 255)

            gtlmark = util.batch_orth_proj(batch['gt_landmark'], batch['cam'])
            gtlmark[..., 1:] = - gtlmark[..., 1:]

            gtlmark = util.tensor_vis_landmarks(batch['gt_image'][visind].unsqueeze(0), gtlmark[visind].unsqueeze(0))
            gtlmark = gtlmark.squeeze(0)
            gtlmark = tensor_util.tensor2im(gtlmark  , normalize = False)
            gtlmark = np.ascontiguousarray(gtlmark, dtype=np.uint8)
            gtlmark = util.writeText(gtlmark, batch['image_path'][visind])
            gtlmark = np.ascontiguousarray(gtlmark, dtype=np.uint8)
            gtlmark = np.clip(gtlmark, 0, 255)

            genimage = predicted_images.data[visind].cpu() #  * self.stdtex + self.meantex 
            genimage = tensor_util.tensor2im(genimage  , normalize = False)
            genimage = np.ascontiguousarray(genimage, dtype=np.uint8)
            genimage = tensor_util.writeText(genimage, batch['image_path'][visind])
            genimage = np.ascontiguousarray(genimage, dtype=np.uint8)
            genimage = np.clip(genimage, 0, 255)

            genlmark = util.batch_orth_proj(landmarks3d, batch['cam'].to(self.device))
            genlmark[..., 1:] = - genlmark[..., 1:]

            genlmark = util.tensor_vis_landmarks(batch['gt_image'].to(self.device)[visind].unsqueeze(0),genlmark[visind].unsqueeze(0))
            genlmark = genlmark.squeeze(0)
            genlmark = tensor_util.tensor2im(genlmark  , normalize = False)
            genlmark = np.ascontiguousarray(genlmark, dtype=np.uint8)
            genlmark = util.writeText(genlmark, batch['image_path'][visind])
            genlmark = np.ascontiguousarray(genlmark, dtype=np.uint8)
            genlmark = np.clip(genlmark, 0, 255)

            visuals = OrderedDict([
            ('gtimage', gtimage),
            ('gtlmark', gtlmark ),
            ('genimage', genimage),
            ('genlmark', genlmark )
            ])
            self.visualizer.display_current_results(visuals, step, 1) 

    def debug(self):
        for p in self.latent2code.parameters():
            p.requires_grad = False 
        for step, batch in enumerate(tqdm(self.data_loader)):
            with torch.no_grad():    
                shape_latent = batch['shape_latent'].to(self.device)
                appearance_latent = batch['appearance_latent'].to(self.device)
                cam, pose = batch['cam'].to(self.device), batch['pose'].to(self.device)

                shape_fea = self.latent2code.Latent2ShapeExpCode(shape_latent)
                shapecode = self.latent2code.latent2shape(shape_fea)
                expcode = self.latent2code.latent2exp(shape_fea)

                app_fea = self.latent2code.Latent2AlbedoLitCode(appearance_latent)
                albedocode = self.latent2code.latent2albedo(app_fea)
                litcode = self.latent2code.latent2lit(app_fea).view(shape_latent.shape[0], 9,3)

                # flame from synthesized shape, exp, lit, albedo
                vertices, landmarks2d, landmarks3d = self.latent2code.flame(shape_params=shapecode, expression_params=expcode, pose_params=pose)
                trans_vertices = util.batch_orth_proj(vertices, cam)
                trans_vertices[..., 1:] = - trans_vertices[..., 1:]

                ## render
                albedos = self.latent2code.flametex(albedocode, self.latent2code.image_size) / 255.
                ops = self.latent2code.render(vertices, trans_vertices, albedos, litcode)
                predicted_images = ops['images']
                
                # flame from sudo ground truth shape, exp, lit, albedo
                recons_vertices, recons_landmarks2d, recons_landmarks3d = self.latent2code.flame(
                                                shape_params = batch['shape'].to(self.device), 
                                                expression_params = batch['exp'].to(self.device),
                                                pose_params=batch['pose'].to(self.device))
                recons_trans_vertices = util.batch_orth_proj(recons_vertices, batch['cam'].to(self.device))
                recons_trans_vertices[..., 1:] = -recons_trans_vertices[..., 1:]

                ## render
                recons_albedos = self.latent2code.flametex(batch['tex'].to(self.device), self.latent2code.image_size) / 255.
                recons_ops = self.latent2code.render(recons_vertices, recons_trans_vertices, recons_albedos, batch['lit'].view(-1,9,3).to(self.device))
                recons_images = recons_ops['images']

            losses = {}
            losses['landmark'] = util.l2_distance(landmarks3d[:, 17:, :2], batch['gt_landmark'][:, 17:, :2].to(self.device)) * self.flame_config.w_lmks
            losses['photometric_texture'] = (batch['img_mask'].to(self.device) * (predicted_images - batch['gt_image'].to(self.device) ).abs()).mean() * self.flame_config.w_pho
            loss = losses['landmark'] + losses['photometric_texture']
            
            tqdm_dict = {'loss_landmark': losses['landmark'].data, 'loss_tex': losses['photometric_texture'].data  }
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in tqdm_dict.items()} 
            self.visualizer.print_current_errors(0, step, errors, 0)

            visind = 0
            gtimage = batch['gt_image'].data[visind].cpu()
            gtimage = tensor_util.tensor2im(gtimage  , normalize = False)
            gtimage = np.ascontiguousarray(gtimage, dtype=np.uint8)
            gtimage = tensor_util.writeText(gtimage, batch['image_path'][visind])
            gtimage = np.ascontiguousarray(gtimage, dtype=np.uint8)
            gtimage = np.clip(gtimage, 0, 255)

            gtlmark = util.batch_orth_proj(batch['gt_landmark'], batch['cam'])
            gtlmark[..., 1:] = - gtlmark[..., 1:]

            gtlmark = util.tensor_vis_landmarks(batch['gt_image'][visind].unsqueeze(0), gtlmark[visind].unsqueeze(0))
            gtlmark = gtlmark.squeeze(0)
            gtlmark = tensor_util.tensor2im(gtlmark  , normalize = False)
            gtlmark = np.ascontiguousarray(gtlmark, dtype=np.uint8)
            gtlmark = util.writeText(gtlmark, batch['image_path'][visind])
            gtlmark = np.ascontiguousarray(gtlmark, dtype=np.uint8)
            gtlmark = np.clip(gtlmark, 0, 255)

            genimage = predicted_images.data[visind].cpu() #  * self.stdtex + self.meantex 
            genimage = tensor_util.tensor2im(genimage  , normalize = False)
            genimage = np.ascontiguousarray(genimage, dtype=np.uint8)
            genimage = tensor_util.writeText(genimage, batch['image_path'][visind])
            genimage = np.ascontiguousarray(genimage, dtype=np.uint8)
            genimage = np.clip(genimage, 0, 255)

            reconsimage = recons_images.data[visind].cpu() #  * self.stdtex + self.meantex 
            reconsimage = tensor_util.tensor2im(reconsimage  , normalize = False)
            reconsimage = np.ascontiguousarray(reconsimage, dtype=np.uint8)
            reconsimage = tensor_util.writeText(reconsimage, batch['image_path'][visind])
            reconsimage = np.ascontiguousarray(reconsimage, dtype=np.uint8)
            reconsimage = np.clip(reconsimage, 0, 255)

            
            genlmark = util.batch_orth_proj(landmarks3d, batch['cam'].to(self.device))
            genlmark[..., 1:] = - genlmark[..., 1:]

            genlmark = util.tensor_vis_landmarks(batch['gt_image'].to(self.device)[visind].unsqueeze(0),genlmark[visind].unsqueeze(0))
            genlmark = genlmark.squeeze(0)
            genlmark = tensor_util.tensor2im(genlmark  , normalize = False)
            genlmark = np.ascontiguousarray(genlmark, dtype=np.uint8)
            genlmark = util.writeText(genlmark, batch['image_path'][visind])
            genlmark = np.ascontiguousarray(genlmark, dtype=np.uint8)
            genlmark = np.clip(genlmark, 0, 255)

            visuals = OrderedDict([
            ('gtimage', gtimage),
            ('gtlmark', gtlmark ),
            ('genimage', genimage),
            ('reconsimage', reconsimage),
            ('genlmark', genlmark )
            ])
            self.visualizer.display_current_results(visuals, step, 1) 
           
def vis_tensor(image_tensor = None, image_path = None, land_tensor = None, cam = None,  visind =0, device = torch.device("cuda")):
    if land_tensor is not None:
        lmark = util.batch_orth_proj(land_tensor.to(device), cam.to(device))
        lmark[..., 1:] = - lmark[..., 1:]
        lmark = util.tensor_vis_landmarks(image_tensor.to(device)[visind].unsqueeze(0),lmark[visind].unsqueeze(0))
        output = lmark.squeeze(0)
    else:
        output = image_tensor.data[visind].cpu() #  * self.stdtex + self.meantex 
    output = tensor_util.tensor2im(output  , normalize = False)
    output = np.ascontiguousarray(output, dtype=np.uint8)
    output = util.writeText(output, image_path)
    output = np.ascontiguousarray(output, dtype=np.uint8)
    output = np.clip(output, 0, 255)

    return output


class Latent2CodeModule():
    def __init__(self, flame_config, opt ):
        super().__init__()
        self.opt = opt
        self.flame_config = flame_config
        self.visualizer = Visualizer(opt)
        if opt.cuda:
            self.device = torch.device("cuda")
        self.latent2code = Latent2Code( flame_config, opt)
        
        self.optimizer =  optim.Adam(self.latent2code.parameters(),lr= self.opt.lr , betas=(self.opt.beta1, 0.999))
        # self.optimizer = optim.Adam( list(self.latent2code.Latent2ShapeExpCode.parameters()) + \
        #                           list(self.latent2code.Latent2AlbedoLitCode.parameters()) + \
        #                           list(self.latent2code.latent2shape.parameters()) + \
        #                           list(self.latent2code.latent2exp.parameters()) + \
        #                           list(self.latent2code.latent2albedo.parameters()) + \
        #                           list(self.latent2code.latent2lit.parameters()) \
        #                           , lr= self.opt.lr , betas=(self.opt.beta1, 0.999))
        if opt.isTrain:
            self.latent2code =torch.nn.DataParallel(self.latent2code, device_ids=range(len(self.opt.gpu_ids)))
        self.latent2code = self.latent2code.to(self.device)
        self.dataset  = FFHQDataset(opt)
        if opt.isTrain:
            self.data_loader = DataLoaderWithPrefetch(self.dataset, \
                        batch_size=opt.batchSize,\
                        drop_last=True,\
                        shuffle = True,\
                        num_workers = opt.nThreads, \
                        prefetch_size = min(8, opt.nThreads))
        else:
            self.data_loader = DataLoaderWithPrefetch(self.dataset, \
                        batch_size=opt.batchSize,\
                        drop_last=False,\
                        shuffle = False,\
                        num_workers = opt.nThreads, \
                        prefetch_size = min(8, opt.nThreads))

        print ('========', len(self.data_loader),'========')
        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.ckpt_path, exist_ok = True)

    def train(self):
        for p in self.latent2code.parameters():
            p.requires_grad = True 
        for epoch in range( 100000):
            for step, batch in enumerate(tqdm(self.data_loader)):
                
                landmarks3d, predicted_images, recons_images = self.latent2code.forward(
                            batch['shape_latent'].to(self.device),
                            batch['appearance_latent'].to(self.device),
                            batch['cam'].to(self.device), 
                            batch['pose'].to(self.device),
                            batch['shape'].to(self.device),
                            batch['exp'].to(self.device),
                            batch['tex'].to(self.device),
                            batch['lit'].to(self.device))
                losses = {}
                losses['landmark'] = util.l2_distance(landmarks3d[:, 17:, :2], batch['gt_landmark'][:, 17:, :2].to(self.device)) * self.flame_config.w_lmks
                losses['photometric_texture'] = (batch['img_mask'].to(self.device) * (predicted_images - batch['gt_image'].to(self.device) ).abs()).mean() * self.flame_config.w_pho
                loss = losses['landmark'] + losses['photometric_texture']
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tqdm_dict = {'loss_landmark': losses['landmark'].data, 'loss_tex': losses['photometric_texture'].data  }
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in tqdm_dict.items()} 
                self.visualizer.print_current_errors(epoch, step, errors, 0)

            if epoch % self.opt.save_step == 0:  
                
                visind = 0
                gtimage = batch['gt_image'].data[0].cpu()
                gtimage = tensor_util.tensor2im(gtimage  , normalize = False)
                gtimage = np.ascontiguousarray(gtimage, dtype=np.uint8)
                gtimage = tensor_util.writeText(gtimage, batch['image_path'][0])
                gtimage = np.ascontiguousarray(gtimage, dtype=np.uint8)
                gtimage = np.clip(gtimage, 0, 255)

                gtlmark = util.batch_orth_proj(batch['gt_landmark'], batch['cam'])
                gtlmark[..., 1:] = - gtlmark[..., 1:]

                gtlmark = util.tensor_vis_landmarks(batch['gt_image'][visind].unsqueeze(0), gtlmark[visind].unsqueeze(0))
                gtlmark = gtlmark.squeeze(0)
                gtlmark = tensor_util.tensor2im(gtlmark  , normalize = False)
                gtlmark = np.ascontiguousarray(gtlmark, dtype=np.uint8)
                gtlmark = util.writeText(gtlmark, batch['image_path'][0])
                gtlmark = np.ascontiguousarray(gtlmark, dtype=np.uint8)
                gtlmark = np.clip(gtlmark, 0, 255)

                genimage = predicted_images.data[0].cpu() #  * self.stdtex + self.meantex 
                genimage = tensor_util.tensor2im(genimage  , normalize = False)
                genimage = np.ascontiguousarray(genimage, dtype=np.uint8)
                genimage = tensor_util.writeText(genimage, batch['image_path'][0])
                genimage = np.ascontiguousarray(genimage, dtype=np.uint8)
                genimage = np.clip(genimage, 0, 255)

                reconsimage = recons_images.data[0].cpu() #  * self.stdtex + self.meantex 
                reconsimage = tensor_util.tensor2im(reconsimage  , normalize = False)
                reconsimage = np.ascontiguousarray(reconsimage, dtype=np.uint8)
                reconsimage = tensor_util.writeText(reconsimage, batch['image_path'][0])
                reconsimage = np.ascontiguousarray(reconsimage, dtype=np.uint8)
                reconsimage = np.clip(reconsimage, 0, 255)

                genlmark = util.batch_orth_proj(landmarks3d, batch['cam'].to(self.device))
                genlmark[..., 1:] = - genlmark[..., 1:]

                genlmark = util.tensor_vis_landmarks(batch['gt_image'].to(self.device)[visind].unsqueeze(0),genlmark[visind].unsqueeze(0))
                genlmark = genlmark.squeeze(0)
                genlmark = tensor_util.tensor2im(genlmark  , normalize = False)
                genlmark = np.ascontiguousarray(genlmark, dtype=np.uint8)
                genlmark = util.writeText(genlmark, batch['image_path'][0])
                genlmark = np.ascontiguousarray(genlmark, dtype=np.uint8)
                genlmark = np.clip(genlmark, 0, 255)

                visuals = OrderedDict([
                ('gtimage', gtimage),
                ('gtlmark', gtlmark ),
                ('genimage', genimage),
                ('reconsimage', reconsimage),
                ('genlmark', genlmark )
                ])
        
                self.visualizer.display_current_results(visuals, epoch, self.opt.save_step) 

                torch.save(self.latent2code.module.Latent2ShapeExpCode.state_dict(), self.opt.Latent2ShapeExpCode_weight)
                torch.save(self.latent2code.module.Latent2AlbedoLitCode.state_dict(),self.opt.Latent2AlbedoLitCode_weight)
                torch.save(self.latent2code.module.latent2shape.state_dict(), self.opt.latent2shape_weight)
                torch.save(self.latent2code.module.latent2exp.state_dict(), self.opt.latent2exp_weight)
                torch.save(self.latent2code.module.latent2albedo.state_dict(), self.opt.latent2albedo_weight)
                torch.save(self.latent2code.module.latent2lit.state_dict(),self.opt.latent2lit_weight)
    def test(self):
        for p in self.latent2code.parameters():
            p.requires_grad = False 
        for step, batch in enumerate(tqdm(self.data_loader)):
            with torch.no_grad():    
                landmarks3d, predicted_images = self.latent2code.forward(
                        batch['shape_latent'].to(self.device), \
                        batch['appearance_latent'].to(self.device), \
                        batch['cam'].to(self.device), batch['pose'].to(self.device))
            losses = {}
            losses['landmark'] = util.l2_distance(landmarks3d[:, 17:, :2], batch['gt_landmark'][:, 17:, :2].to(self.device)) * self.flame_config.w_lmks
            losses['photometric_texture'] = (batch['img_mask'].to(self.device) * (predicted_images - batch['gt_image'].to(self.device) ).abs()).mean() * self.flame_config.w_pho
            loss = losses['landmark'] + losses['photometric_texture']
            
            tqdm_dict = {'loss_landmark': losses['landmark'].data, 'loss_tex': losses['photometric_texture'].data  }
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in tqdm_dict.items()} 
            self.visualizer.print_current_errors(0, step, errors, 0)

            visind = 0
            gtimage = batch['gt_image'].data[visind].cpu()
            gtimage = tensor_util.tensor2im(gtimage  , normalize = False)
            gtimage = np.ascontiguousarray(gtimage, dtype=np.uint8)
            gtimage = tensor_util.writeText(gtimage, batch['image_path'][visind])
            gtimage = np.ascontiguousarray(gtimage, dtype=np.uint8)
            gtimage = np.clip(gtimage, 0, 255)

            gtlmark = util.batch_orth_proj(batch['gt_landmark'], batch['cam'])
            gtlmark[..., 1:] = - gtlmark[..., 1:]

            gtlmark = util.tensor_vis_landmarks(batch['gt_image'][visind].unsqueeze(0), gtlmark[visind].unsqueeze(0))
            gtlmark = gtlmark.squeeze(0)
            gtlmark = tensor_util.tensor2im(gtlmark  , normalize = False)
            gtlmark = np.ascontiguousarray(gtlmark, dtype=np.uint8)
            gtlmark = util.writeText(gtlmark, batch['image_path'][visind])
            gtlmark = np.ascontiguousarray(gtlmark, dtype=np.uint8)
            gtlmark = np.clip(gtlmark, 0, 255)

            genimage = predicted_images.data[visind].cpu() #  * self.stdtex + self.meantex 
            genimage = tensor_util.tensor2im(genimage  , normalize = False)
            genimage = np.ascontiguousarray(genimage, dtype=np.uint8)
            genimage = tensor_util.writeText(genimage, batch['image_path'][visind])
            genimage = np.ascontiguousarray(genimage, dtype=np.uint8)
            genimage = np.clip(genimage, 0, 255)

            genlmark = util.batch_orth_proj(landmarks3d, batch['cam'].to(self.device))
            genlmark[..., 1:] = - genlmark[..., 1:]

            genlmark = util.tensor_vis_landmarks(batch['gt_image'].to(self.device)[visind].unsqueeze(0),genlmark[visind].unsqueeze(0))
            genlmark = genlmark.squeeze(0)
            genlmark = tensor_util.tensor2im(genlmark  , normalize = False)
            genlmark = np.ascontiguousarray(genlmark, dtype=np.uint8)
            genlmark = util.writeText(genlmark, batch['image_path'][visind])
            genlmark = np.ascontiguousarray(genlmark, dtype=np.uint8)
            genlmark = np.clip(genlmark, 0, 255)

            visuals = OrderedDict([
            ('gtimage', gtimage),
            ('gtlmark', gtlmark ),
            ('genimage', genimage),
            ('genlmark', genlmark )
            ])
            self.visualizer.display_current_results(visuals, step, 1) 

    def debug(self):
        for p in self.latent2code.parameters():
            p.requires_grad = False 
        for step, batch in enumerate(tqdm(self.data_loader)):
            with torch.no_grad():    
                shape_latent = batch['shape_latent'].to(self.device)
                appearance_latent = batch['appearance_latent'].to(self.device)
                cam, pose = batch['cam'].to(self.device), batch['pose'].to(self.device)

                shape_fea = self.latent2code.Latent2ShapeExpCode(shape_latent)
                shapecode = self.latent2code.latent2shape(shape_fea)
                expcode = self.latent2code.latent2exp(shape_fea)

                app_fea = self.latent2code.Latent2AlbedoLitCode(appearance_latent)
                albedocode = self.latent2code.latent2albedo(app_fea)
                litcode = self.latent2code.latent2lit(app_fea).view(shape_latent.shape[0], 9,3)

                # flame from synthesized shape, exp, lit, albedo
                vertices, landmarks2d, landmarks3d = self.latent2code.flame(shape_params=shapecode, expression_params=expcode, pose_params=pose)
                trans_vertices = util.batch_orth_proj(vertices, cam)
                trans_vertices[..., 1:] = - trans_vertices[..., 1:]

                ## render
                albedos = self.latent2code.flametex(albedocode, self.latent2code.image_size) / 255.
                ops = self.latent2code.render(vertices, trans_vertices, albedos, litcode)
                predicted_images = ops['images']
                
                # flame from sudo ground truth shape, exp, lit, albedo
                recons_vertices, recons_landmarks2d, recons_landmarks3d = self.latent2code.flame(
                                                shape_params = batch['shape'].to(self.device), 
                                                expression_params = batch['exp'].to(self.device),
                                                pose_params=batch['pose'].to(self.device))
                recons_trans_vertices = util.batch_orth_proj(recons_vertices, batch['cam'].to(self.device))
                recons_trans_vertices[..., 1:] = -recons_trans_vertices[..., 1:]

                ## render
                recons_albedos = self.latent2code.flametex(batch['tex'].to(self.device), self.latent2code.image_size) / 255.
                recons_ops = self.latent2code.render(recons_vertices, recons_trans_vertices, recons_albedos, batch['lit'].view(-1,9,3).to(self.device))
                recons_images = recons_ops['images']

            losses = {}
            losses['landmark'] = util.l2_distance(landmarks3d[:, 17:, :2], batch['gt_landmark'][:, 17:, :2].to(self.device)) * self.flame_config.w_lmks
            losses['photometric_texture'] = (batch['img_mask'].to(self.device) * (predicted_images - batch['gt_image'].to(self.device) ).abs()).mean() * self.flame_config.w_pho
            loss = losses['landmark'] + losses['photometric_texture']
            
            tqdm_dict = {'loss_landmark': losses['landmark'].data, 'loss_tex': losses['photometric_texture'].data  }
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in tqdm_dict.items()} 
            self.visualizer.print_current_errors(0, step, errors, 0)

            visind = 0
            gtimage = batch['gt_image'].data[visind].cpu()
            gtimage = tensor_util.tensor2im(gtimage  , normalize = False)
            gtimage = np.ascontiguousarray(gtimage, dtype=np.uint8)
            gtimage = tensor_util.writeText(gtimage, batch['image_path'][visind])
            gtimage = np.ascontiguousarray(gtimage, dtype=np.uint8)
            gtimage = np.clip(gtimage, 0, 255)

            gtlmark = util.batch_orth_proj(batch['gt_landmark'], batch['cam'])
            gtlmark[..., 1:] = - gtlmark[..., 1:]

            gtlmark = util.tensor_vis_landmarks(batch['gt_image'][visind].unsqueeze(0), gtlmark[visind].unsqueeze(0))
            gtlmark = gtlmark.squeeze(0)
            gtlmark = tensor_util.tensor2im(gtlmark  , normalize = False)
            gtlmark = np.ascontiguousarray(gtlmark, dtype=np.uint8)
            gtlmark = util.writeText(gtlmark, batch['image_path'][visind])
            gtlmark = np.ascontiguousarray(gtlmark, dtype=np.uint8)
            gtlmark = np.clip(gtlmark, 0, 255)

            genimage = predicted_images.data[visind].cpu() #  * self.stdtex + self.meantex 
            genimage = tensor_util.tensor2im(genimage  , normalize = False)
            genimage = np.ascontiguousarray(genimage, dtype=np.uint8)
            genimage = tensor_util.writeText(genimage, batch['image_path'][visind])
            genimage = np.ascontiguousarray(genimage, dtype=np.uint8)
            genimage = np.clip(genimage, 0, 255)

            reconsimage = recons_images.data[visind].cpu() #  * self.stdtex + self.meantex 
            reconsimage = tensor_util.tensor2im(reconsimage  , normalize = False)
            reconsimage = np.ascontiguousarray(reconsimage, dtype=np.uint8)
            reconsimage = tensor_util.writeText(reconsimage, batch['image_path'][visind])
            reconsimage = np.ascontiguousarray(reconsimage, dtype=np.uint8)
            reconsimage = np.clip(reconsimage, 0, 255)


            genlmark = util.batch_orth_proj(landmarks3d, batch['cam'].to(self.device))
            genlmark[..., 1:] = - genlmark[..., 1:]

            genlmark = util.tensor_vis_landmarks(batch['gt_image'].to(self.device)[visind].unsqueeze(0),genlmark[visind].unsqueeze(0))
            genlmark = genlmark.squeeze(0)
            genlmark = tensor_util.tensor2im(genlmark  , normalize = False)
            genlmark = np.ascontiguousarray(genlmark, dtype=np.uint8)
            genlmark = util.writeText(genlmark, batch['image_path'][visind])
            genlmark = np.ascontiguousarray(genlmark, dtype=np.uint8)
            genlmark = np.clip(genlmark, 0, 255)

            visuals = OrderedDict([
            ('gtimage', gtimage),
            ('gtlmark', gtlmark ),
            ('genimage', genimage),
            ('reconsimage', reconsimage),
            ('genlmark', genlmark )
            ])
            self.visualizer.display_current_results(visuals, step, 1) 
           