import os.path
from PIL import Image, ImageChops, ImageFile
import PIL
import json
import pickle 
import cv2
import numpy as np
import random
import torch
from tqdm import tqdm
import  os, time
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
from torch.utils.data.dataloader import (
    _SingleProcessDataLoaderIter,
    _MultiProcessingDataLoaderIter,
)


class DataLoaderWithPrefetch(DataLoader):
    def __init__(self, *args, prefetch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch_size = (
            prefetch_size
            if prefetch_size is not None
            else 2 * kwargs.get("num_workers", 0)
        )

    def __iter__(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIterWithPrefetch(self)


class _MultiProcessingDataLoaderIterWithPrefetch(_MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        self.prefetch_size = loader.prefetch_size
        self._tasks_outstanding=self.prefetch_size
        super().__init__(loader)
        
    def _reset(self, loader, first_iter=False):
        super(_MultiProcessingDataLoaderIter, self)._reset(loader, first_iter)
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        self._task_info = {}
        self._workers_status = [True for i in range(self._num_workers)]
        if not first_iter:
            for idx in range(self._num_workers):
                self._index_queues[idx].put(_utils.worker._ResumeIteration())
            resume_iteration_cnt = self._num_workers
            while resume_iteration_cnt > 0:
                data = self._get_data()
                if isinstance(data, _utils.worker._ResumeIteration):
                    resume_iteration_cnt -= 1
        # print(f"im fetching {self.prefetch_size}")
        for _ in range(self.prefetch_size):
            self._try_put_index()


class FFHQDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt

        if opt.isTrain:
            list_path = os.path.join(opt.dataroot, "ffhq_trainlist.pkl")
            zip_path = os.path.join(opt.dataroot, 'ffhq_train.pkl' )
        else:
            list_path = os.path.join(opt.dataroot, "ffhq_testlist.pkl")
            zip_path = os.path.join(opt.dataroot, 'ffhq_test.pkl' )

        if opt.debug:
            list_path = list_path[:-4] + '_debug.pkl'
            zip_path = zip_path[:-4] + '_debug.pkl'
        
        _file = open(list_path, "rb")
        self.data_list = pickle.load(_file)
        _file.close()

        _file = open(zip_path, "rb")
        self.total_data = pickle.load(_file)
        _file.close()

        transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        if opt.debug:
            self.data_list = self.data_list[:opt.datanum]
            for i in range(opt.datanum):
                name = self.data_list[i]
                img_path = os.path.join(self.opt.dataroot, 'images',name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                maskimg_path = os.path.join(self.opt.dataroot, 'imagemasks',name[:-3] +'npy')
                self.total_data[name]['img_mask'] = np.load(maskimg_path)

                self.total_data[name]['gt_image'] = self.transform(img)
                self.total_data[name]['image_path'] = name

        
        print ('******************', len(self.data_list), len(self.total_data))
        self.total_t = []
    """
            data[name] ={'shape': shape, 
                 'exp': exp,
                 'pose': pose,
                 'cam': cam,
                 'tex': tex,
                 'lit': lit,
                 'cam_pose': camera_pose,
                 'z_nerf': z_nerf,
                 'z_gan': z_gan,
                 'gt_img': img,
                 'gt_landmark': landmark
                 #'img_mask':image_masks
                }
        """
    def __getitem__(self, index):
        
        name = self.data_list[index]
        data = self.total_data[name]
        # data = copy.copy(self.total_data[name])
        if not self.opt.debug:
            # img_path = os.path.join(self.opt.dataroot, 'images',name)
            # img = cv2.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # maskimg_path = os.path.join(self.opt.dataroot, 'imagemasks',name[:-3] +'npy')
            # data['img_mask'] = np.load(maskimg_path)

            # data['gt_image'] = self.transform(img)
            data['image_path'] = name
        return data

    def __len__(self):
        return len(self.data_list) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FFHQDataset'


class FFHQRigDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt

        if opt.isTrain:
            list_path = os.path.join(opt.dataroot, "ffhq_trainlist.pkl")
            zip_path = os.path.join(opt.dataroot, 'ffhq_train.pkl' )
        else:
            list_path = os.path.join(opt.dataroot, "ffhq_trainlist.pkl")
            zip_path = os.path.join(opt.dataroot, 'ffhq_train.pkl' )

        if opt.debug:
            list_path = list_path[:-4] + '_debug.pkl'
            zip_path = zip_path[:-4] + '_debug.pkl'
        
        _file = open(list_path, "rb")
        self.data_list = pickle.load(_file)
        _file.close()

        if opt.debug:
            self.data_list = self.data_list[:opt.datanum]

        _file = open(zip_path, "rb")
        self.total_data = pickle.load(_file)
        _file.close()

        transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

        print ('******************', len(self.data_list), len(self.total_data))
        self.total_t = []
    def __getitem__(self, index):

        name = self.data_list[index]
        data = copy.copy(self.total_data[self.data_list[index]])
        """
            data[name] ={'shape': shape, 
                 'exp': exp,
                 'pose': pose,
                 'cam': cam,
                 'tex': tex,
                 'lit': lit,
                 'cam_pose': camera_pose,
                 'z_nerf': z_nerf,
                 'z_gan': z_gan,
                 'gt_img': img,
                 'gt_landmark': landmark
                 #'img_mask':image_masks
                }
        """
        img_path = os.path.join(self.opt.dataroot, 'images',name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        maskimg_path = os.path.join(self.opt.dataroot, 'imagemasks',name[:-3] +'npy')
        data['img_mask'] = np.load(maskimg_path)

        data['gt_image'] = self.transform(img)
        data['image_path'] = name

        another_inx = torch.randint(0, self.__len__() ,(1,)).item()
        name2 = self.data_list[another_inx]
        data2 = copy.copy(self.total_data[self.data_list[another_inx]])

        img_path2 = os.path.join(self.opt.dataroot, 'images',name2)
        img2 = cv2.imread(img_path2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        maskimg_path2 = os.path.join(self.opt.dataroot, 'imagemasks',name2[:-3] +'npy')
        data2['img_mask'] = np.load(maskimg_path2)

        data2['gt_image'] = self.transform(img2)
        data2['image_path'] = name2

        return [data, data2]

    def __len__(self):
        return len(self.data_list) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FFHQRigDataset'