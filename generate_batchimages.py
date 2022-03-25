# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
import time
import glob
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import imageio
import legacy
from renderer import Renderer
import cv2
#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------
os.environ['PYOPENGL_PLATFORM'] = 'egl'

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', default='/nfs/STG/CodecAvatar/lelechen/FFHQ/generated_stylenerf/checkpoint/ffhq_1024.pkl')
@click.option('--startseeds', type=int, help='start of random seeds', default =10000)
@click.option('--endseeds', type=int, help='end of random seeds', default =100000)

@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.5, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, default='/nfs/STG/CodecAvatar/lelechen/FFHQ/generated_stylenerf')
@click.option('--name', help='exepriment name Where to save the output images', type=str, default='images')

@click.option('--render-program', default=None, show_default=True)
@click.option('--render-option', default=None, type=str, help="e.g. up_256, camera, depth")

def generate_images(
    ctx: click.Context,
    network_pkl: str,
    startseeds: int,
    endseeds: int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    name: str,
    render_program=None,
    render_option=None,
):

    device = torch.device('cuda')
    if os.path.isdir(network_pkl):
        network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % network_pkl)
    
    with dnnlib.util.open_url(network_pkl) as f:
        network = legacy.load_network_pkl(f)
        G = network['G_ema'].to(device) # type: ignore
        D = network['D'].to(device)
    # from fairseq import pdb;pdb.set_trace()
    os.makedirs(outdir +f'/{name}', exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    
    # avoid persistent classes... 
    from training.networks import Generator
    # from training.stylenerf import Discriminator
    from torch_utils import misc
    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
    G2 = Renderer(G2, D, program=render_program, withws = True)
    
    # Generate images.
    all_imgs = []

    def stack_imgs(imgs):
        img = torch.stack(imgs, dim=2)
        return img.reshape(img.size(0) * img.size(1), img.size(2) * img.size(3), 3)

    def proc_img(img): 
        return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

   
    for seed_idx, seed in enumerate(range(startseeds, endseeds)):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, endseeds))
        G2.set_random_seed(seed)
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        relative_range_u = [0.5]
        img, ws = G2(
            z=z,
            c=label,
            truncation_psi=truncation_psi,
            noise_mode=noise_mode,
            render_option=render_option,
            n_steps=1,
            relative_range_u=relative_range_u,
            return_cameras=True)
        
        img = proc_img(img)[0]
        np.save(  f'{outdir}/stylecode/w/{seed:0>6d}.npy' ,  ws[0].detach().cpu().numpy() )
        PIL.Image.fromarray(img.numpy(), 'RGB').save(f'{outdir}/{name}/{seed:0>6d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
