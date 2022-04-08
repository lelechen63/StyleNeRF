import numpy as np
import os
import ntpath
import time
import sys
sys.path.append('/home/uss00022/lelechen/github/CIPS-3D')
import util
from utils import html_

import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.use_html = True # opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if not opt.isTrain:
            self.name +='_test'
        

        if not os.path.exists( os.path.join( opt.checkpoints_dir, self.name )):
            os.mkdir(os.path.join( opt.checkpoints_dir, self.name ))
        print ('############################################')
        print (os.path.join( opt.checkpoints_dir, self.name ))

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, self.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, self.name, 'loss_log.txt')
        with open(self.log_name, "w") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        
        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
               
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.jpg' % (epoch, label))
                print (img_path)
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html_.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30)
            for n in range(epoch, 0, -step):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.jpg' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

   

  
    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t1, t2, t3):
        message = '(epoch: %d, iters: %d, data time: %.3f, network time: %.3f,loss time: %.3f, ) ' % (epoch, i, t1, t2, t3)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.6f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        print (image_path, 'img path')
        image_dir = webpage.get_image_dir()
        print (image_dir, 'img dir')
        # short_path = ntpath.basename(image_path[0])
        tmp = image_path[0].split('/')
        short_path = tmp[-3] +"__" +tmp[-2] +'__' +tmp[-1]
        print (short_path, 'short path')
        name = os.path.splitext(short_path)[0]
        print (name, 'name')
        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (name, label)
            print (image_name, 'image_name')

            save_path = os.path.join(image_dir, image_name)

            print (save_path, 'save_path')
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
