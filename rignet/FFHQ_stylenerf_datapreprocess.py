import os
import pickle 

import numpy as np 
from tqdm import tqdm
import cv2


def get_train( debug = False):
    root_p = '/nfs/STG/CodecAvatar/lelechen/FFHQ/generated_stylenerf'    
    img_lists = sorted(os.listdir( os.path.join(root_p, 'images') )) 
    if debug:
        img_lists = img_lists[:200]
    traindata = {}
    testdata = {}
    traintest_threashhold = int(0.8*len(img_lists))
    for it, name in enumerate(tqdm(img_lists)):
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

            landmark = np.squeeze(flame_p['landmark3d'], axis=0) #[1,68,2]
            """ 
                we normalize the landmark into 0-1.
                landmark[:, 0] = landmark[:, 0] / float(image.shape[2]) * 2 - 1
                landmark[:, 1] = landmark[:, 1] / float(image.shape[1]) * 2 - 1
            """
            z = np.load( os.path.join( root_p, 'stylecode', 'w', str(k) +'.npy' ) ).reshape(-1)   #[17*512]
            if it < traintest_threashhold:
                traindata[name] ={'shape': shape, 
                        'exp': exp,
                        'pose': pose,
                        'cam': cam,
                        'tex': tex,
                        'lit': lit,
                        'latent': z,
                        'gt_landmark': landmark
                        }
            else:
                testdata[name] ={'shape': shape, 
                        'exp': exp,
                        'pose': pose,
                        'cam': cam,
                        'tex': tex,
                        'lit': lit,
                        'latent': z,
                        'gt_landmark': landmark
                        }
        else:
            print (flame_path, 'Does not exist!!!')
    print (len(traindata), len(testdata))
    if debug:
        with open( os.path.join(root_p, "ffhq_train_debug.pkl" ), 'wb') as handle:
            pickle.dump(traindata, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open( os.path.join(root_p, "ffhq_trainlist_debug.pkl" ), 'wb') as handle:
            pickle.dump(img_lists[:traintest_threashhold], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open( os.path.join(root_p, "ffhq_test_debug.pkl" ), 'wb') as handle:
            pickle.dump(testdata, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open( os.path.join(root_p, "ffhq_trainlist_debug.pkl" ), 'wb') as handle:
            pickle.dump(img_lists[traintest_threashhold:], handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open( os.path.join(root_p, "ffhq_train.pkl" ), 'wb') as handle:
            pickle.dump(traindata, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open( os.path.join(root_p, "ffhq_trainlist.pkl" ), 'wb') as handle:
            pickle.dump(img_lists[:traintest_threashhold], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open( os.path.join(root_p, "ffhq_test.pkl" ), 'wb') as handle:
            pickle.dump(testdata, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open( os.path.join(root_p, "ffhq_trainlist.pkl" ), 'wb') as handle:
            pickle.dump(img_lists[traintest_threashhold:], handle, protocol=pickle.HIGHEST_PROTOCOL)

get_train(debug = True)