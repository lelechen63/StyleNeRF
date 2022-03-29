import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics        
        self.parser.add_argument('--cuda', type=str, default='cuda', help='cuda/cpu')        
        self.parser.add_argument('--name', type=str, default='Latent2Code', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--fp16', action='store_true', default=False, help='train with AMP')
        self.parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
        self.parser.add_argument('--datanum', type=int, default=32, help='number of samples for training')

        self.parser.add_argument('--datasetname', type=str, default='ffhq')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=8 , help='input batch size')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='/nfs/STG/CodecAvatar/lelechen/FFHQ/generated_stylenerf/') 
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data') 
        self.parser.add_argument('--isTrain', action='store_false', help='isTrain is for training')                
        self.parser.add_argument('--meannorm', action='store_true', help='weight for feature matching loss')          
        self.parser.add_argument('--modeltype', type=int, default=2, help='number of clusters for features')        

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.model = self.opt.name
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        # if len(self.opt.gpu_ids) > 0:
        #     torch.cuda.set_device(self.opt.gpu_ids[0])
        
        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        os.makedirs(expr_dir, exist_ok = True )
        latent2code_dir = os.path.join(self.opt.checkpoints_dir, "Latent2Code")
        self.opt.Latent2ShapeExpCode_weight = os.path.join(latent2code_dir,'Latent2ShapeExpCode.pth' )
        self.opt.Latent2AlbedoLitCode_weight = os.path.join(latent2code_dir,'Latent2AlbedoLitCode.pth' )
        self.opt.latent2shape_weight = os.path.join(latent2code_dir,'latent2shape.pth' )
        self.opt.latent2exp_weight = os.path.join(latent2code_dir,'latent2exp.pth' )
        self.opt.latent2albedo_weight = os.path.join(latent2code_dir,'latent2albedo.pth' )
        self.opt.latent2lit_weight = os.path.join(latent2code_dir,'latent2lit.pth' )
        # self.opt.latent2pose_weight = os.path.join(latent2code_dir,'latent2pose.pth' )

        self.opt.WEncoder_weight = os.path.join(expr_dir,'WEncoder_weight.pth' )
        self.opt.ParamEncoder_weight = os.path.join(expr_dir,'ParamEncoder_weight.pth' )
        self.opt.WDecoder_weight = os.path.join(expr_dir,'WDecoder_weight.pth' )
        
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        
            
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
