import os
import pickle
import glob
import torch
import pdb
import h5py
from tqdm import tqdm
import numpy as np

from collections import namedtuple

DiffusionExperiment_Feb19 = namedtuple('DiffusionExperiment_Feb19', 'dataset gcp_model video_model ema trainer epoch')

def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False

def get_latest_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), 'model-*.pt')
    latest_epoch = -1
    for state in states:
        if state == 'model-front.pt':
            continue
        epoch = int(state.replace('model-', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    # print(config)
    return config
