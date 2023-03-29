import sys

# extend PATH
sys.path.extend([
    'src/taming-transformers',
    '/home/kieran/iarts/stable_diff/deforum/venv_deforum/src/taming-transformers',
    '/home/kieran/iarts/stable_diff/deforum/venv_deforum/src/clip',
    'src/clip',
    'stable-diffusion/',
    'k-diffusion',
    'pytorch3d-lite',
    'AdaBins',
    'MiDaS',
    'utils'
])

from utils import *
import py3d_tools as p3d
from helpers import DepthModel, sampler_fn
from k_diffusion.external import CompVisDenoiser
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import argparse
import subprocess
import os
import json
from IPython import display

import gc, math, os, pathlib, subprocess, sys, time
import cv2
import numpy as np
import pandas as pd
import random
import re
import requests
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from skimage.exposure import match_histograms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from types import SimpleNamespace
from torch import autocast

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", help="select cuda device")
parser.add_argument("-m", "--max_frames", help="total frames in animation")
parser.add_argument("-p", "--prompt", help="prompt for stable diff generation")
cli_args = parser.parse_args()
max_frames = int(cli_args.max_frames)
prompt = cli_args.prompt

# Get all available GPUs
sub_p_res = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], stdout=subprocess.PIPE).stdout.decode('utf-8')
print(sub_p_res)

# Set rank for single GPU
os.environ['CUDA_VISIBLE_DEVICES'] = cli_args.device

# create dir for models and outputs if they don't exist
models_path = "./content/models" #@param {type:"string"}
output_path = "./content/output" #@param {type:"string"}

os.makedirs(models_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

print(f"models_path: {models_path}")
print(f"output_path: {output_path}")

# Stable Diffusion model config
model_config = "v1-inference.yaml" #@param ["custom","v1-inference.yaml"]
model_checkpoint =  "sd-v1-4.ckpt" #@param ["custom","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt"]
custom_config_path = "" #@param {type:"string"}
custom_checkpoint_path = "" #@param {type:"string"}

load_on_run_all = True #@param {type: 'boolean'}
half_precision = True # check
check_sha256 = True #@param {type:"boolean"}

model_map = {
    "sd-v1-4-full-ema.ckpt": {'sha256': '14749efc0ae8ef0329391ad4436feb781b402f4fece4883c7ad8d10556d8a36a'},
    "sd-v1-4.ckpt": {'sha256': 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556'},
    "sd-v1-3-full-ema.ckpt": {'sha256': '54632c6e8a36eecae65e36cb0595fab314e1a1545a65209f24fde221a8d4b2ca'},
    "sd-v1-3.ckpt": {'sha256': '2cff93af4dcc07c3e03110205988ff98481e86539c51a8098d4f2236e41f7f2f'},
    "sd-v1-2-full-ema.ckpt": {'sha256': 'bc5086a904d7b9d13d2a7bccf38f089824755be7261c7399d92e555e1e9ac69a'},
    "sd-v1-2.ckpt": {'sha256': '3b87d30facd5bafca1cbed71cfb86648aad75d1c264663c0cc78c7aea8daec0d'},
    "sd-v1-1-full-ema.ckpt": {'sha256': 'efdeb5dc418a025d9a8cc0a8617e106c69044bc2925abecc8a254b2910d69829'},
    "sd-v1-1.ckpt": {'sha256': '86cd1d3ccb044d7ba8db743d717c9bac603c4043508ad2571383f954390f3cea'}
}

# config path
ckpt_config_path = custom_config_path if model_config == "custom" else os.path.join(models_path, model_config)
if os.path.exists(ckpt_config_path):
    print(f"{ckpt_config_path} exists")
else:
    ckpt_config_path = "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
print(f"Using config: {ckpt_config_path}")

# checkpoint path or download
ckpt_path = custom_checkpoint_path if model_checkpoint == "custom" else os.path.join(models_path, model_checkpoint)
ckpt_valid = True
if os.path.exists(ckpt_path):
    print(f"{ckpt_path} exists")
else:
    print(f"Please download model checkpoint and place in {os.path.join(models_path, model_checkpoint)}")
    ckpt_valid = False

if check_sha256 and model_checkpoint != "custom" and ckpt_valid:
    import hashlib
    print("\n...checking sha256")
    with open(ckpt_path, "rb") as f:
        bytes = f.read() 
        hash = hashlib.sha256(bytes).hexdigest()
        del bytes
    if model_map[model_checkpoint]["sha256"] == hash:
        print("hash is correct\n")
    else:
        print("hash in not correct\n")
        ckpt_valid = False

if ckpt_valid:
    print(f"Using ckpt: {ckpt_path}")

if load_on_run_all and ckpt_valid:
    local_config = OmegaConf.load(f"{ckpt_config_path}")
    model = load_model_from_config(local_config, f"{ckpt_path}", half_precision=half_precision)
    device = torch.device("cuda")
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

# PROMPT GENERATION
with open('base_prompts.txt', 'r') as file:
    base_prompts = file.read()

generation_prompt = f"{{rococo {prompt} headdress | {prompt} cornucopia | macro {prompt} | {prompt} Cthulhu | {prompt} nuclear explosion | {prompt} mushroom cloud | Hubble {prompt} nebula | {prompt} infestation | steampunk {prompt} | magic rubber {prompt} | psychedelic {prompt} | {prompt} couture}}"
base_prompt = base_prompts.replace("{x}", generation_prompt)

# CHECK DIFF WITH motion["keyframe_frames"] from generate_motion()
key_frames = list(np.arange(max_frames)[::100])
key_frames = [x + np.random.randint(-20, 20) for x in key_frames]
key_frames[0] = 0

animation_prompts = {}
for kf in key_frames:
    new_prompt = pick_variant(base_prompts)
    new_prompt = " ".join(new_prompt.split())
    
    animation_prompts[int(kf)] = new_prompt

# Run 
if __name__ == '__main__':

    args = SimpleNamespace(**DeforumArgs(output_path, prompt))
    anim_args = SimpleNamespace(**DeforumAnimArgs(max_frames))

    args.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))

    if args.seed == -1:
        args.seed = random.randint(0, 2**32 - 1)
    if not args.use_init:
        args.init_image = None
    if args.sampler == 'plms' and (args.use_init or anim_args.animation_mode != 'None'):
        print(f"Init images aren't supported with PLMS yet, switching to KLMS")
        args.sampler = 'klms'
    if args.sampler != 'ddim':
        args.ddim_eta = 0

    if anim_args.animation_mode == 'None':
        anim_args.max_frames = 1
    elif anim_args.animation_mode == 'Video Input':
        args.use_init = True

    # clean up unused memory
    gc.collect()
    torch.cuda.empty_cache()

    # dispatch to appropriate renderer
    if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
        render_animation(device, args, half_precision, model, anim_args, models_path, animation_prompts)
    elif anim_args.animation_mode == 'Video Input':
        render_input_video(args, anim_args)
    elif anim_args.animation_mode == 'Interpolation':
        render_interpolation(device, args, model, anim_args, animation_prompts)
    else:
        render_image_batch(args)  

    images_folder = os.path.join(output_path, time.strftime('%Y-%m'), args.batch_name)

    render_video(images_folder, f"{args.timestring}_{args.prompt}")

    print("Generation finished")

        