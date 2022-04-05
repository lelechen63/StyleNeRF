
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
        self.optimizer = optim.Adam( list(self.rig.LatentEncoder.parameters()) + \
                                  list(self.rig.ParamEncoder.parameters()) + \
                                   list(self.rig.LatentDecoder.parameters())\
                                  , lr= self.opt.lr , betas=(self.opt.beta1, 0.999))
        for p in self.rig.Latent2fea.parameters():
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
        l2loss = nn.MSELoss()
        t0 = time.time()
        for epoch in range( 10000):
            for step, batch in enumerate(tqdm(self.data_loader)):
                t1 = time.time()
                return_list = self.rig.forward(
                    batch[0]['latent'].to(self.device),
                    batch[1]['latent'].to(self.device),
                    
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
                latent_w_same = return_list['latent_w_same'] 
                landmark_w_ = return_list['landmark_w_']
                render_img_w_ = return_list['render_img_w_']
                landmark_v_ = return_list['landmark_v_'] 
                render_img_v_ = return_list['render_img_v_']

                t2 = time.time()
                losses = {}
                # keep batch[1], w the same
                # losses['landmark_same'] = util.l2_distance(landmark_same[:, 17:, :2], batch[1]['gt_landmark'][:, 17:, :2].to(self.device)) * self.flame_config.w_lmks
                # losses['photometric_texture_same'] = (batch[1]['img_mask'].to(self.device) * (render_img_same - batch[1]['gt_image'].to(self.device) ).abs()).mean() * self.flame_config.w_pho
                losses['w_same'] = l2loss(latent_w_same,batch[1]['latent'].to(self.device) )
                # close to w
                losses['landmark_w_'] = util.l2_distance(landmark_w_[:, 17:, :2], batch[1]['gt_landmark'][:, 17:, :2].to(self.device)) * self.flame_config.w_lmks
                losses['photometric_texture_w_'] = l2loss(batch[1]['img_mask'].to(self.device) * render_img_w_,  batch[1]['img_mask'].to(self.device) * batch[1]['gt_image'].to(self.device) ) * self.flame_config.w_pho
                
                # close to v
                losses['landmark_v_'] = util.l2_distance(landmark_v_[:, 17:, :2], batch[0]['gt_landmark'][:, 17:, :2].to(self.device)) * self.flame_config.w_lmks
                losses['photometric_texture_v_'] = l2loss(batch[0]['img_mask'].to(self.device) * render_img_v_,  batch[0]['img_mask'].to(self.device) * batch[0]['gt_image'].to(self.device) ) * self.flame_config.w_pho

                loss = losses['w_same'] + \
                       losses['landmark_w_'] + losses['photometric_texture_w_'] + \
                       losses['landmark_v_'] + losses['photometric_texture_v_']
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tqdm_dict = {'w_same': losses['w_same'].data, \
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
                                        image_path = batch[0]['image_path'][0] +'---V-gtimg',
                                        device = self.device
                                         )

                lmark_v = vis_tensor(image_tensor= batch[0]['gt_image'], 
                                        image_path = batch[0]['image_path'][0] +'--V-landmark',
                                        land_tensor = batch[0]['gt_landmark'],
                                        cam = batch[0]['cam'], 
                                        device = self.device
                                         )
               

                image_w = vis_tensor(image_tensor= batch[1]['gt_image'], 
                                        image_path = batch[1]['image_path'][0] +'---W-gtimg',
                                        device = self.device
                                         )

                lmark_w = vis_tensor(image_tensor= batch[1]['gt_image'], 
                                        image_path = batch[1]['image_path'][0] +'--W-gtlandmrk',
                                        land_tensor = batch[1]['gt_landmark'],
                                        cam = batch[1]['cam'], 
                                        device = self.device
                                         )

                recons_images_w = vis_tensor(image_tensor= recons_images_w, 
                                        image_path = batch[1]['image_path'][0] +'---recons-W-img',
                                        device = self.device
                                         )
                recons_images_v = vis_tensor(image_tensor= recons_images_v, 
                                        image_path = batch[0]['image_path'][0] +'---recons-V-img',
                                        device = self.device
                                         )

                genlmark_same = vis_tensor(image_tensor= batch[1]['gt_image'], 
                                        image_path = batch[1]['image_path'][0] +'---same-W-landamrk',
                                        land_tensor = landmark_same,
                                        cam = batch[1]['cam'], 
                                        device = self.device
                                         )
        
                genimage_same = vis_tensor(image_tensor= render_img_same, 
                                        image_path = batch[1]['image_path'][0] +'---same-W-renderimg',
                                        device = self.device
                                         )
          
                genlmark_w = vis_tensor(image_tensor= batch[1]['gt_image'], 
                                        image_path = batch[1]['image_path'][0] +'---close-W-landmark',
                                        land_tensor = landmark_w_,
                                        cam = batch[1]['cam'], 
                                        device = self.device
                                         )

                genimage_w = vis_tensor(image_tensor= render_img_w_, 
                                        image_path = batch[1]['image_path'][0] +'---close-W-renderimg',
                                        device = self.device
                                         )

                genlmark_v = vis_tensor(image_tensor= batch[0]['gt_image'], 
                                        image_path = batch[0]['image_path'][0] +'---close-V-landmark',
                                        land_tensor = landmark_v_,
                                        cam = batch[0]['cam'], 
                                        device = self.device
                                         )
                genimage_v = vis_tensor(image_tensor = render_img_v_, 
                                        image_path = batch[0]['image_path'][0]+'---close-V-renderimg', 
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

                torch.save(self.rig.module.LatentEncoder.state_dict(), self.opt.WEncoder_weight)
                torch.save(self.rig.module.ParamEncoder.state_dict(),self.opt.ParamEncoder_weight)
                torch.save(self.rig.module.LatentDecoder.state_dict(), self.opt.WDecoder_weight)
               
    def test(self):
        for p in self.rig.parameters():
            p.requires_grad = False 
    
        choice_dic =["shape", "exp", "albedo", "lit"]
        for step, batch in enumerate(tqdm(self.data_loader)):
            with torch.no_grad():    
                
                return_list = self.rig.test(
                            batch[0]['latent'].to(self.device),
                            batch[1]['latent'].to(self.device),
                            
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
                
                landmark_same = return_list['landmark_same']
                render_img_same =  return_list['render_img_same']
                landmark_w_= return_list['landmark_w_']
                render_img_w_ = return_list['render_img_w_']
                landmark_v_ = return_list['landmark_v_']
                render_img_v_ = return_list['render_img_v_']
                recons_images_v = return_list['recons_images_v']
                recons_images_w = return_list['recons_images_w']
                choice = return_list['choice']
                syns_v = return_list['syns_v']
                syns_w = return_list['syns_w']
                syns_w_same = return_list['syns_w_same']
                syns_w_hat = return_list['syns_w_hat']
                recons_images_w_hat = return_list['recons_images_hat']

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
                
                tqdm_dict = {'landmark_same': losses['landmark_same'].data, \
                             'photometric_texture_same': losses['photometric_texture_same'].data, \
                             'landmark_w_': losses['landmark_w_'].data, \
                             'photometric_texture_w_': losses['photometric_texture_w_'].data, \
                             'landmark_v_': losses['landmark_v_'].data, \
                             'photometric_texture_v_': losses['photometric_texture_v_'].data
                               }
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in tqdm_dict.items()} 
                self.visualizer.print_current_errors(0, step, errors, 0,0,0)
                t0 = time.time()
                
                # visualize the image close to v
                image_v = vis_tensor(image_tensor= batch[0]['gt_image'], 
                                        image_path = batch[0]['image_path'][0] +'-V-gtimg',
                                        device = self.device
                                         )

                lmark_v = vis_tensor(image_tensor= batch[0]['gt_image'], 
                                        image_path = '-V-landmark',
                                        land_tensor = batch[0]['gt_landmark'],
                                        cam = batch[0]['cam'], 
                                        device = self.device
                                         )
               
                image_w = vis_tensor(image_tensor= batch[1]['gt_image'], 
                                        image_path = batch[1]['image_path'][0] +'-W-gtimg',
                                        device = self.device
                                         )

                lmark_w = vis_tensor(image_tensor= batch[1]['gt_image'], 
                                        image_path = 'W-gtlandmrk',
                                        land_tensor = batch[1]['gt_landmark'],
                                        cam = batch[1]['cam'], 
                                        device = self.device
                                         )

                recons_images_w = vis_tensor(image_tensor= recons_images_w, 
                                        image_path = '-recons-W-img',
                                        device = self.device
                                         )
                recons_images_v = vis_tensor(image_tensor= recons_images_v, 
                                        image_path = 'recons-V-img',
                                        device = self.device
                                         )

                genlmark_same = vis_tensor(image_tensor= batch[1]['gt_image'], 
                                        image_path = 'same-W-landamrk',
                                        land_tensor = landmark_same,
                                        cam = batch[1]['cam'], 
                                        device = self.device
                                         )
        
                genimage_same = vis_tensor(image_tensor= render_img_same, 
                                        image_path = 'same-W-renderimg',
                                        device = self.device
                                         )
          
                genlmark_w = vis_tensor(image_tensor= batch[1]['gt_image'], 
                                        image_path = 'close-W-landmark',
                                        land_tensor = landmark_w_,
                                        cam = batch[1]['cam'], 
                                        device = self.device
                                         )

                genimage_w = vis_tensor(image_tensor= render_img_w_, 
                                        image_path = 'close-W-renderimg',
                                        device = self.device
                                         )

                genlmark_v = vis_tensor(image_tensor= batch[0]['gt_image'], 
                                        image_path = 'close-V-landmark',
                                        land_tensor = landmark_v_,
                                        cam = batch[0]['cam'], 
                                        device = self.device
                                         )
                genimage_v = vis_tensor(image_tensor = render_img_v_, 
                                        image_path = 'close-V-renderimg', 
                                        device = self.device)

                synsimg_v = vis_ganimg(image_tensor= syns_v, 
                                        image_path = 'V-syns',
                                         )

                synsimg_w = vis_ganimg(image_tensor= syns_w, 
                                        image_path = '-W-syns',
                                         )

                synsimg_w_same = vis_ganimg(image_tensor= syns_w_same, 
                                        image_path = 'w-syns-same',
                                         )

                synsimg_w_hat = vis_ganimg(image_tensor= syns_w_hat, 
                                        image_path ='-W-hat-' + choice_dic[choice],
                                         )
                recons_images_w_hat = vis_tensor(image_tensor= recons_images_w_hat, 
                                        image_path = 'recons-w-hat',
                                        device = self.device
                                         )

                visuals = OrderedDict([
                ('image_v', image_v),
                ('lmark_v', lmark_v),
                ('recons_images_v', recons_images_v),

                ('image_w', image_w),
                ('lmark_w', lmark_w),
                ('recons_images_w', recons_images_w),
                
                ('genlmark_same_W', genlmark_same ),
                ('genlmark_v', genlmark_v),
                ('genimage_v', genimage_v ),
                ('genimage_same_W', genimage_same),
                ('genlmark_w', genlmark_w),
                ('genimage_w', genimage_w ),

                ('synsimg_v', synsimg_v ),
                ('synsimg_w', synsimg_w ),
                ('synsimg_w_same', synsimg_w_same ),
                ('synsimg_w_hat', synsimg_w_hat ),
                ('recons_w_hat', recons_images_w_hat ),
                ])
        
                self.visualizer.display_current_results(visuals, step, self.opt.save_step) 
  

def vis_ganimg(image_tensor = None, image_path = None):
   
    output = (image_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()[0]
    output = np.ascontiguousarray(output, dtype=np.uint8)
    output = util.writeText(output, image_path)
    output = np.ascontiguousarray(output, dtype=np.uint8)
    output = np.clip(output, 0, 255)

    return output


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

