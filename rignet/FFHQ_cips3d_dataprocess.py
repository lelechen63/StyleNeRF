import os
import pickle 

import numpy as np 
from tqdm import tqdm
import cv2

def debug_single():
    with open("/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg/flame_p.pickle", 'rb') as f:
        flame_p = pickle.load(f, encoding='latin1')

    for key in flame_p.keys():
        print ('=-------------')
        print (key, flame_p[key].shape )
        # print (flame_p[key])

    with open("/home/uss00022/lelechen/github/CIPS-3D/results/model_interpolation/gt.pkl", 'rb') as handle:
        info = pickle.load(handle)
    info = info['results/model_interpolation/0.png']
    for key in info.keys():
        print ('=-------------')
        try:
            print (key, info[key].shape )
        except:
            print (key, info[key] )


# debug_single()

def get_debug():
    with open("/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg/flame_p.pickle", 'rb') as f:
        flame_p = pickle.load(f, encoding='latin1')

    with open("/home/uss00022/lelechen/github/CIPS-3D/results/model_interpolation/gt.pkl", 'rb') as handle:
        info = pickle.load(handle)
    
    name = '0.png'
    info = info['results/model_interpolation/0.png']
    img_p = '/home/uss00022/lelechen/github/CIPS-3D/results/model_interpolation/0.png'
    img = cv2.imread(img_p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    shape = flame_p['shape'].reshape(-1) #[1,100]
    exp = flame_p['exp'].reshape(-1) #[1,50]
    pose = flame_p['pose'].reshape(-1) #[1,6]
    cam = flame_p['cam'].reshape(-1) #[1,3]
    tex = flame_p['tex'].reshape(-1) #[1,50]
    lit = flame_p['lit'].reshape(-1) #[1,9,3]
    print(flame_p['image_masks'].shape, '+++++')
    image_masks = np.squeeze(flame_p['image_masks'],axis=0)
    
    landmark = np.squeeze(flame_p['landmark3d'], axis=0) #[1,68,2]
    """ 
        we normalize the landmark into 0-1.
        landmark[:, 0] = landmark[:, 0] / float(image.shape[2]) * 2 - 1
        landmark[:, 1] = landmark[:, 1] / float(image.shape[1]) * 2 - 1
    """
    camera_pose = info['cur_camera_pos'].reshape(-1) #[1,3]
    z_nerf = info['z_nerf'].reshape(-1) # [1,256]
    z_gan = info['z_inr'].reshape(-1) #[1,512]

    data = {}
    ffhq_trainlist = []

    ffhq_trainlist.append(name)
    data[name] ={'shape': shape, 
                 'exp': exp,
                 'pose': pose,
                 'cam': cam,
                 'tex': tex,
                 'lit': lit,
                 'cam_pose': camera_pose,
                 'shape_latent': z_nerf,
                 'appearance_latent': z_gan,
                 'gt_img': img,
                 'gt_landmark': landmark,
                 'img_mask':image_masks
                }
    with open("/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg/ffhq_train_debug.pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("/home/uss00022/lelechen/github/CIPS-3D/photometric_optimization/gg/ffhq_trainlist_debug.pkl", 'wb') as handle:
        pickle.dump(ffhq_trainlist, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_flame_total(root_p, k, debug = False):
    if debug:
        k = 10
    total_flame = {}
    for i in range(0,k):
        flame_p = os.path.join(root_p, 'flame', str(i), 'flame_p.pickle')
        if os.path.exists(flame_p):
            with open(flame_p, 'rb') as f:
                flamelist = pickle.load(f, encoding='latin1')
            total_flame[str(i)] = flamelist
        else:
            print (flame_p,' not exists!!')
    print (len(total_flame))
    total_flame_p = os.path.join(root_p, 'flame', 'total_flame.pickle')
    if debug:
        total_flame_p = total_flame_p[:-7] +'_debug.pickle'
    with open(total_flame_p, 'wb') as handle:
        pickle.dump(total_flame, handle, protocol=pickle.HIGHEST_PROTOCOL)

# def get_cips_total():

def get_train(k = 200000, debug = False):
    root_p = '/nfs/STG/CodecAvatar/lelechen/FFHQ/generated_cips3d'    
    with open(os.path.join(root_p, 'images', 'z_info.pkl'), 'rb') as handle:
        z_p = pickle.load(handle)
    print ( len(z_p))
    # print (flame_p.keys()) # key: 1
    # print (z_p.keys()) # key: 1.png

    data = {}
    ffhq_trainlist = []

    for name in tqdm(z_p.keys()):
        k = name[:-4]
        flame_path = os.path.join(root_p, 'flame', str(k), 'flame_p.pickle')
        if os.path.exists(flame_path):
            with open(flame_path, 'rb') as f:
                flame_p = pickle.load(f, encoding='latin1')
            shape = flame_p['shape'].reshape(-1) #[1,100]
            exp = flame_p['exp'].reshape(-1) #[1,50]
            pose = flame_p['pose'].reshape(-1) #[1,6]
            cam = flame_p['cam'].reshape(-1) #[1,3]
            tex = flame_p['tex'].reshape(-1) #[1,50]
            lit = flame_p['lit'].reshape(-1) #[1,9,3]

            imgmask_path = os.path.join(root_p, 'imagemasks', k+'.npy' )
            if not os.path.exists(imgmask_path):
                image_masks = np.squeeze(flame_p['image_masks'],axis=0)
                np.save(imgmask_path, image_masks)

            landmark = np.squeeze(flame_p['landmark3d'], axis=0) #[1,68,2]
            """ 
                we normalize the landmark into 0-1.
                landmark[:, 0] = landmark[:, 0] / float(image.shape[2]) * 2 - 1
                landmark[:, 1] = landmark[:, 1] / float(image.shape[1]) * 2 - 1
            """

            info = z_p[name]
            # camera_pose = info['cur_camera_pos'].reshape(-1) #[1,3]
            z_nerf = info['z_nerf'].reshape(-1) # [1,256]
            z_gan = info['z_inr'].reshape(-1) #[1,512]

            ffhq_trainlist.append(name)
            data[name] ={'shape': shape, 
                        'exp': exp,
                        'pose': pose,
                        'cam': cam,
                        'tex': tex,
                        'lit': lit,
                        # 'cam_pose': camera_pose,
                        'shape_latent': z_nerf,
                        'appearance_latent': z_gan,
                        # 'gt_img': img,
                        'gt_landmark': landmark,
                        # 'img_mask':image_masks
                        }
        else:
            print (flame_path, 'Does not exist!!!')
    print (len(data), len(ffhq_trainlist))
    with open( os.path.join(root_p, "ffhq_train.pkl" ), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open( os.path.join(root_p, "ffhq_trainlist.pkl" ), 'wb') as handle:
        pickle.dump(ffhq_trainlist, handle, protocol=pickle.HIGHEST_PROTOCOL)

get_train(debug = False)