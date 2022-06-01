
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from model import *
sys.path.append('/home/us000218/lelechen/github/CIPS-3D/utils')
from visualizer import Visualizer
import util
from dataset import *
import time
import imageio
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()



class CodeAutoEncoderModule():
    def __init__(self, flame_config, opt ):
        super().__init__()
        self.opt = opt
        self.flame_config = flame_config
        self.visualizer = Visualizer(opt)
        self.l2_loss = nn.MSELoss()
        self.MSE_Loss   = nn.SmoothL1Loss(reduction='mean')
        if opt.cuda:
            self.device = torch.device("cuda")
        self.autoencoder = CodeAutoEncoder( flame_config, opt)
        
        self.optimizer = optim.Adam( list(self.autoencoder.encoder.parameters()) + \
                                  list(self.autoencoder.decoder.parameters()) + \
                                  , lr= self.opt.lr , betas=(self.opt.beta1, 0.999))
        for p in self.latent2code.flame.parameters():
            p.requires_grad = False 
        for p in self.latent2code.flametex.parameters():
            p.requires_grad = False 
        
        if opt.isTrain:
            self.latent2code =torch.nn.DataParallel(self.latent2code, device_ids=range(len(self.opt.gpu_ids)))
        self.latent2code = self.latent2code.to(self.device)
        if opt.name == 'Latent2Code':
            if opt.inversefit:
                self.dataset = FFHQRigDataset(opt)
            else:
                self.dataset  = FFHQDataset(opt)
    
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

        self.litmean =  torch.Tensor(np.load(opt.dataroot + '/litmean.npy')).to('cuda')
        self.expmean = torch.Tensor(np.load(opt.dataroot + '/expmean.npy')).to('cuda')
        self.shapemean = torch.Tensor(np.load(opt.dataroot + '/shapemean.npy')).to('cuda')
        self.albedomean = torch.Tensor(np.load(opt.dataroot + '/albedomean.npy')).to('cuda') 

    def train(self):
        t0 = time.time()
        
        iteration = 0

        for epoch in range( 100000):
            for step, batch in enumerate(tqdm(self.data_loader)):
                t1 = time.time()

                for key in batch.keys():
                    if key !='image_path':
                        batch[key] = batch[key].to(self.device)

                #landmarks3d, predicted_images, recons_images 
                return_list = self.latent2code.forward(
                            batch['latent'],
                            batch['cam'], 
                            batch['pose']
                            )
                losses = {}
                t2 = time.time()
                expcode, shapecode, litcode, albedocode = return_list['expcode'], return_list['shapecode'], return_list['litcode'], return_list['albedocode']
                loss = 0
                if self.opt.supervision =='render':
                    landmarks3d, predicted_images  = return_list['landmarks3d'], return_list['predicted_images'].float()
                    losses['landmark'] = util.l2_distance(landmarks3d[:, :, :2], batch['gt_landmark'][:, :, :2]) * self.flame_config.w_lmks
                    losses['photometric_texture'] = self.MSE_Loss( batch['img_mask'] * predicted_images ,  batch['img_mask'] * batch['gt_image']) * self.flame_config.w_pho                    
                    loss = losses['landmark'] + losses['photometric_texture'] #+ losses['lit_reg'] + losses['albedo_reg'] + losses['expression_reg'] + losses['shape_reg']
                else:
                    losses['expcode'] = self.l2_loss(expcode, batch['exp'])
                    losses['shapecode'] = self.l2_loss(shapecode, batch['shape'])
                    losses['litcode'] = self.l2_loss(litcode, batch['lit'])
                    losses['albedocode'] = self.l2_loss(albedocode, batch['tex'])
                    # losses['pose'] = self.l2_loss(posecode, batch['pose'])
                    for key in losses.keys():
                        loss += losses[key]
                

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                t3 = time.time()
                tqdm_dict ={}
                for key in losses.keys():
                    tqdm_dict[key] = losses[key].data
                
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in tqdm_dict.items()} 
                self.visualizer.print_current_errors(epoch, step, errors, t1-t0, t2-t1, t3-t2 )
                t0 = time.time()
                if iteration % self.opt.save_step == 0:
                    return_list = self.latent2code.forward(
                            batch['latent'],
                            batch['cam'], 
                            batch['pose'],
                            batch['shape'],
                            batch['exp'],
                            batch['tex'],
                            batch['lit'])

                    visind = 0
                
                    genimage = vis_tensor(image_tensor= return_list['predicted_images'], 
                                            image_path = batch['image_path'][0] ,
                                            device = self.device
                                            )
                    
                    reconsimage = vis_tensor(image_tensor= return_list['recons_images'], 
                                            image_path = batch['image_path'][0],
                                            device = self.device
                                            )
                    gtlmark = vis_tensor(image_tensor= return_list['recons_images'], 
                                            image_path = batch['image_path'][0],
                                            land_tensor = batch['gt_landmark'],
                                            cam = batch['cam'], 
                                            device = self.device
                                            )
                    genlmark = vis_tensor(image_tensor= return_list['predicted_images'], 
                                            image_path = batch['image_path'][0],
                                            land_tensor = return_list['landmarks3d'],
                                            cam = batch['cam'], 
                                            device = self.device
                                            )
                    if self.opt.supervision =='render':
                        gtimage = vis_tensor(image_tensor= batch['gt_image'], 
                                            image_path = batch['image_path'][0] ,
                                            device = self.device
                                            )
                        visuals = OrderedDict([
                        ('gtimage', gtimage),
                        ('gtlmark', gtlmark ),
                        ('genimage', genimage),
                        ('reconsimage', reconsimage),
                        ('genlmark', genlmark )
                        ])
                    else:                   
                        visuals = OrderedDict([
                        ('gtlmark', gtlmark ),
                        ('genimage', genimage),
                        ('reconsimage', reconsimage),
                        ('genlmark', genlmark )
                        ])

                    self.visualizer.display_current_results(visuals, iteration, self.opt.save_step) 
                    torch.save(self.latent2code.module.Latent2fea.state_dict(), self.opt.Latent2ShapeExpCode_weight)
                    torch.save(self.latent2code.module.latent2shape.state_dict(), self.opt.latent2shape_weight)
                    torch.save(self.latent2code.module.latent2exp.state_dict(), self.opt.latent2exp_weight)
                    torch.save(self.latent2code.module.latent2albedo.state_dict(), self.opt.latent2albedo_weight)
                    torch.save(self.latent2code.module.latent2lit.state_dict(),self.opt.latent2lit_weight)
                    # torch.save(self.latent2code.module.latent2pose.state_dict(),self.opt.latent2lit_weight)
                iteration +=1
    def test(self):
        for p in self.latent2code.parameters():
            p.requires_grad = False 
        for step, batch in enumerate(tqdm(self.data_loader)):
            with torch.no_grad():
                for key in batch.keys():
                    if key !='image_path':
                        batch[key] = batch[key].to(self.device)
                return_list = self.latent2code.forward(
                            batch['latent'],
                            batch['cam'], 
                            batch['pose'],
                            batch['shape'],
                            batch['exp'],
                            batch['tex'],
                            batch['lit'])
                losses = {}
                if self.opt.supervision =='render':
                    landmarks3d, predicted_images  = return_list['landmarks3d'], return_list['predicted_images']
                    
                    losses['landmark'] = util.l2_distance(landmarks3d[:, :, :2], batch['gt_landmark'][:, :, :2]) * self.flame_config.w_lmks
                    print (losses['landmark'],']=====')
                    losses['photometric_texture'] = self.MSE_Loss( batch['img_mask'] * predicted_images ,  batch['img_mask'] * batch['gt_image']) * self.flame_config.w_pho  
                    loss = losses['landmark'] + losses['photometric_texture']
                else:
                
                    expcode, shapecode, litcode, albedocode  = return_list['expcode'], return_list['shapecode'], return_list['litcode'], return_list['albedocode']
                    losses['expcode'] = self.l2_loss(expcode, batch['exp'])
                    losses['shapecode'] = self.l2_loss(shapecode, batch['shape'])
                    losses['litcode'] = self.l2_loss(litcode, batch['lit'])
                    losses['albedocode'] = self.l2_loss(albedocode, batch['tex'])
                
                loss = 0
                for key in losses.keys():
                    loss += losses[key]

                tqdm_dict ={}
                for key in losses.keys():
                    tqdm_dict[key] = losses[key].data
                
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in tqdm_dict.items()} 
                self.visualizer.print_current_errors(0, step, errors, 0, 0, 0 )          

                genimage = vis_tensor(image_tensor= return_list['predicted_images'], 
                                        image_path = batch['image_path'][0] ,
                                        device = self.device
                                         )
                gtimage = vis_tensor(image_tensor= batch['gt_image'], 
                                        image_path = batch['image_path'][0] ,
                                        device = self.device
                                         )

                reconsimage = vis_tensor(image_tensor= return_list['recons_images'], 
                                        image_path = batch['image_path'][0],
                                        device = self.device
                                         )
                gtlmark = vis_tensor(image_tensor= return_list['recons_images'], 
                                        image_path = batch['image_path'][0],
                                        land_tensor = batch['gt_landmark'],
                                        cam = batch['cam'], 
                                        device = self.device
                                         )
                genlmark = vis_tensor(image_tensor= return_list['predicted_images'], 
                                        image_path = batch['image_path'][0],
                                        land_tensor = return_list['landmarks3d'],
                                        cam = batch['cam'], 
                                        device = self.device
                                         )

            visuals = OrderedDict([
            ('gtimage', gtimage),
            ('gtlmark', gtlmark ),
            ('genimage', genimage),
            ('reconsimage', reconsimage),
            ('genlmark', genlmark )
            ])
            self.visualizer.display_current_results(visuals, step, 1) 

    def inversefit(self):
        layer_num = 1
        for p in self.latent2code.parameters():
            p.requires_grad = False
        
        if not os.path.exists( os.path.join( self.opt.checkpoints_dir, self.opt.name )):
            os.mkdir(os.path.join( self.opt.checkpoints_dir, self.opt.name ))
        outdir = os.path.join( self.opt.checkpoints_dir, self.opt.name )
        for step, batch in enumerate(self.data_loader):

            for key in batch[0].keys():
                if key !='image_path':
                    batch[0][key] = batch[0][key].to(self.device)
                    batch[1][key] = batch[1][key].to(self.device)
            batchw = batch[0]
            batchv = batch[1]
            batch_ = batchv
            latent = batch_['latent'].view(-1,  512)#.repeat(1,21,1).view(-1, 21*512)

            writer = SummaryWriter('./logs')
            # timestamp = time.strftime('%Y%m%d.%H%M%S',time.localtime(time.time()))
            timestamp = '0'
            video = imageio.get_writer(f'{outdir}/proj_{timestamp}.mp4', mode='I', fps=24, codec='libx264', bitrate='16M')
            print (f'Saving optimization progress video "{outdir}/proj.mp4"')

            # initialize the style code
            z_samples = np.random.RandomState(123).randn(1, self.latent2code.latent_dim)
            z_samples = torch.from_numpy(z_samples).to(self.device)
            z_samples = z_samples.type(latent.dtype)
            
            ws = z_samples.clone()
            ws.requires_grad = True
            batch_['img_mask'].requires_grad = False
            batch_['gt_image'].requires_grad = False
            # batch_['pose'].requires_grad = True
            optimizer = optim.Adam([ws], lr=0.05, betas=(0.9,0.999), eps=1e-8)
            iter_num = self.opt.fititer
            for iter in tqdm(range(iter_num)):
                optimizer.zero_grad()
                return_list = self.latent2code.forward(
                            ws.view(-1,512),
                            batch_['cam'], 
                            batch_['pose'],
                            batch_['shape'],
                            batch_['exp'],
                            batch_['tex'],
                            batch_['lit'])

                expcode, shapecode, litcode, albedocode  = return_list['expcode'], return_list['shapecode'], return_list['litcode'], return_list['albedocode']
                # print(expcode.requires_grad)
                # print(shapecode.requires_grad)

                # print(albedocode.requires_grad)
                # print(litcode.requires_grad)
                # print ('======')
                
                
                loss =0.0
                losses = {}
                # losses['expcode'] = self.l2_loss(expcode, batch_['exp'])
                # losses['shapecode'] = self.l2_loss(shapecode, batch_['shape'])
                # losses['litcode'] = self.l2_loss(litcode, batch_['lit'])
                # losses['albedocode'] = self.l2_loss(albedocode, batch_['tex'])
                # losses['syns_img'] = self.l2_loss(  batch_['img_mask'] *  syns_img,  batch_['img_mask'] *  batch_['gt_image'])* self.flame_config.w_pho
                # losses['syns_img'] = losses['syns_img'].type(torch.FloatTensor)
                # flame from synthesized shape, exp, lit, albedo
                # print((shapecode + self.shapemean).requires_grad) 
                # print((expcode + self.expmean).requires_grad) 
                # print(batch_['pose'].requires_grad) 
                # print('********')
                vertices, landmarks2d, landmarks3d = self.latent2code.flame(shape_params=shapecode + self.shapemean , expression_params=expcode + self.expmean, pose_params=batch_['pose'])
                trans_vertices = util.batch_orth_proj(vertices, batch_['cam'])
                trans_vertices[..., 1:] = - trans_vertices[..., 1:]

                # landmarks3d = util.batch_orth_proj(landmarks3d, batch_['cam'])
                # landmarks3d[..., 1:] = - landmarks3d[..., 1:]


                ## render
                albedos = self.latent2code.flametex(albedocode + self.albedomean, self.latent2code.image_size) / 255.
                ops = self.latent2code.render(vertices, trans_vertices, albedos, (batch_['lit'] + self.litmean).view(-1,9,3))
                predicted_images = ops['images'].type(expcode.type())

                # flame from gt shape, exp, lit, albedo
                vertices, landmarks2d, _ = self.latent2code.flame(shape_params=batch_['shape'] + self.shapemean , expression_params=batch_['exp'] + self.expmean, pose_params=batch_['pose'])
                trans_vertices = util.batch_orth_proj(vertices, batch_['cam'])
                trans_vertices[..., 1:] = - trans_vertices[..., 1:]
                # landmarks3d = util.batch_orth_proj(landmarks3d, batch_['cam'])
                # landmarks3d[..., 1:] = - landmarks3d[..., 1:]

                ## render
                albedos = self.latent2code.flametex(batch_['tex'] + self.albedomean, self.latent2code.image_size) / 255.
                ops = self.latent2code.render(vertices, trans_vertices, albedos, (batch_['lit'] + self.litmean).view(-1,9,3))
                gterender = ops['images']

                losses['landmark'] = self.l2_loss(landmarks3d[:, :, :2], batch_['gt_landmark'][:, :, :2]) * self.flame_config.w_lmks
                # losses['landmark'] = losses['landmark'].type(torch.FloatTensor)
                losses['photometric_texture'] = self.l2_loss(batch_['img_mask'] *predicted_images, batch_['img_mask'] * batch_['gt_image']) * self.flame_config.w_pho
                # losses['photometric_texture'] = losses['photometric_texture'].type(torch.FloatTensor)
                # losses['ws'] = torch.mean((ws **2), dim =[1]) * 0.01
                for k in losses.keys():
                    # print (k, losses[k],'=========')
                    writer.add_scalar(k, losses[k], iter)
                    loss += losses[k] 
                loss.backward()
                optimizer.step()
                

                if iter % 100 == 0:
                    print ( 'ews:', ws.max(),ws.min(), ws.mean(), 'gt: ' , batch_['latent'].max(), batch_['latent'].min(), batch_['latent'].mean() )
                
                    # ggg = batch_['latent'].view(-1, 21,512)
                    # ggg[0,:] = ggg[0,0]
                    tmp1 = batch_['img_mask'] * predicted_images
                    tmp2 = batch_['img_mask'] * batch_['gt_image']
                    tmp3 = batch_['img_mask'] * gterender
                    syns_img = self.latent2code.G2.forward(styles = ws.view(-1,1,512).repeat(1,21,1))['img']
                    syns_img = (F.interpolate(syns_img, size=(self.opt.imgsize, self.opt.imgsize), mode='bilinear') + 1 )/2

                    # syns_img = self.latent2code.G2.forward(styles = ws.view(-1,1,512).repeat(1,21,1))['img']
                    recons_img = self.latent2code.G2.forward(styles =latent.view(-1,1,512).repeat(1,21,1))['img'] 
                    # syns_img = (F.interpolate(syns_img, size=(self.opt.imgsize, self.opt.imgsize), mode='bilinear') + 1 )/2
                    recons_img = (F.interpolate(recons_img, size=(self.opt.imgsize, self.opt.imgsize), mode='bilinear')+ 1 )/2
                    noise =  torch.zeros(1, 21 * 512, dtype=torch.float32) +  (0.1**0.5)*torch.randn(1, 21 * 512)
                    noise = noise.to(self.device)
                    recons_img_noise = self.latent2code.G2.forward(styles = (latent.view(-1,1,512).repeat(1,21,1) + noise.view(-1, 21, 512)))['img'] 
                    recons_img_noise = (F.interpolate(recons_img_noise, size=(self.opt.imgsize, self.opt.imgsize), mode='bilinear')+ 1 )/2
                    # print (recons_img.max(),recons_img.min(), batch_['gt_image'].max(),  batch_['gt_image'].min() )
                    image = torch.cat([tmp1, tmp2, tmp3, batch_['gt_image'], predicted_images,syns_img,recons_img, recons_img_noise ], -1)
                    image = image.permute(0, 2, 3, 1) * 255.
                    image = image.clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    video.append_data(image)  

            video.close()
            writer.close()   
    def inversefit2(self):
        for p in self.latent2code.parameters():
            p.requires_grad = False
        
        if not os.path.exists( os.path.join( self.opt.checkpoints_dir, self.opt.name )):
            os.mkdir(os.path.join( self.opt.checkpoints_dir, self.opt.name ))
        outdir = os.path.join( self.opt.checkpoints_dir, self.opt.name )
        for step, batch in enumerate(self.data_loader):

            for key in batch[0].keys():
                if key !='image_path':
                    batch[0][key] = batch[0][key].to(self.device)
                    batch[1][key] = batch[1][key].to(self.device)
            batchw = batch[0] 
            batchv = batch[1]
            batch_ = batchw

            latent = batch_['latent'].view(-1, 1, 512).repeat(1,21,1).view(-1, 21*512)
            
            recons_img = self.latent2code.G2.forward(styles =latent.view(-1, 21,512))['img'] 
            recons_img = (F.interpolate(recons_img, size=(self.opt.imgsize, self.opt.imgsize), mode='bilinear')+ 1 )/2
            noise =  torch.zeros(1, 21 * 512, dtype=torch.float32) +  (0.1**0.5)*torch.randn(1, 21 * 512)
            noise = noise.to(self.device)
            recons_img_noise = self.latent2code.G2.forward(styles = (latent + noise).view(-1, 21,512))['img'] 
            recons_img_noise = (F.interpolate(recons_img_noise, size=(self.opt.imgsize, self.opt.imgsize), mode='bilinear')+ 1 )/2
            # print (recons_img.max(),recons_img.min(), batch_['gt_image'].max(),  batch_['gt_image'].min() )
            image = torch.cat([batch_['gt_image'],recons_img, recons_img_noise ], -1)
            image = image.permute(0, 2, 3, 1) * 255.
            image = image.clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            print(image.shape)
            cv2.imwrite('./tmp/ggg%s.png'%step,  cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
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