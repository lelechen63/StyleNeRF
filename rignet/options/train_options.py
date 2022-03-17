from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # supervision type 
        self.parser.add_argument('--supervision', type=str, default='render', help='supervised by renderred image+landamrk or by flame code')

        # for displays
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        self.parser.add_argument('--save_step', type=int, default=5, help='load the pretrained model from the specified location')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--pretrain', type=bool, default=False, help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
              
        self.parser.add_argument('--lambda_pix', type=float, default=10, help='weight for feature texture loss')                
        self.parser.add_argument('--lambda_mesh', type=float, default=0.01, help='weight for feature mesh loss')                
        self.parser.add_argument('--lambda_code', type=float, default= 1, help='weight for feature code loss')  