from torch.utils.data import Dataset
import torch, random, pdb, copy
from glob import glob
from PIL import Image
import numpy as np
import torchvision.transforms as T
from torchvideotransforms import video_transforms, volume_transforms
from einops import rearrange
import diffuser.utils as utils
from itertools import product
from environment.libero.lb_env_v3 import LiberoEnvList_V3
    
class LB_Online_Dataset(Dataset):
    '''
    dataset to train a goal conditioned policy, 
    init from SequentialDatasetv2
    '''
    def __init__(self,
                 env, # str or env instance
                 target_size=(128, 128),
                #  randomcrop=False,
                 dataset_config={}):
        '''
        Used when sampling video for guided execution
        '''
        utils.print_color(f"Preparing dataset {env} ...")
        
        self.env_list: LiberoEnvList_V3
        self.env_list = env_list = utils.load_environment(env) # 2.would load_env inside
        self.target_size = target_size

        tk_list = env_list.task_list
        cam_list = env_list.camera_list
        assert np.array(cam_list) == np.array(['agent',])
        
        # sd_list = env_list.env_init_states[tk_list[0]]

        self.task_list = copy.deepcopy(tk_list)
        self.cam_list = copy.deepcopy(cam_list)
        self.dataset_config = dataset_config
        self.act_min_max = dataset_config['act_min_max']
        self.action_dim = len(self.act_min_max[0])
        ## we don't need large orn
        assert (self.act_min_max[0][3:6] <= -0.1).all()
        assert self.action_dim == 7

        ## create combination
        self.combo_type = dataset_config['combo_type']
        if self.combo_type == 'all':
            # self.combo = list(product(tk_list, cam_list, sd_list))
            self.combo = []
            for tk in self.task_list:
                for cam in cam_list:
                    for sd in self.env_list.seed_sets[tk]:
                        self.combo.append( (tk, cam, sd) )
            utils.print_color(f'combo: {self.combo}',)

        elif self.combo_type == 'no_cam':
            assert False
        else:
            raise NotImplementedError

        
    
    
    def __len__(self):
        return len(self.combo)
    
    def __getitem__(self, idx):
        if self.combo_type == 'all':
            tk, c_name, env_idx = self.combo[idx]
        elif self.combo_type == 'no_cam':
            tk, env_idx = self.combo[idx]
            c_name = random.choice(self.cam_list) # return an element
            
        # pdb.set_trace()
        return tk, c_name, env_idx
    
    def sample_random_tensor(self, b_size, act_len, device):
        ## in range [0, 1]
        task_strs = random.sample(self.env_list.task_list, b_size) # str
        img1 = torch.rand(b_size, 3, *self.target_size).to(device)
        img2 = torch.rand(b_size, 3, *self.target_size).to(device)
        
        if act_len: # of the model
            act_shape = (b_size, act_len, self.action_dim)
        else:
            act_shape = (b_size, self.action_dim)
        act = torch.rand( *act_shape ).to(device)

        return [ img1, img2, task_strs, act ]

