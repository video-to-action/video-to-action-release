import sys, pdb
sys.path.insert(0, './')
import gym, gc, os, torch, random, yaml, h5py
import numpy as np
from importlib import reload
import environment; # reload(environment)
from environment.libero.lb_env_v3 import LiberoEnvList_V3
from diffuser.utils import utils
from tqdm import tqdm
import warnings
warnings.simplefilter('always', ResourceWarning)  # Show all resource warnings
from tap import Tap
from environment.libero.lb_data.lb_randsam_utils import lb_rand_sample_1_ep

class Parser(Tap):
    sub_conf: str = 'config.maze2d'

def main():
    '''a helper script to generate dataset of random samples in the Libero env'''

    args = Parser().parse_args()

    file_path = 'environment/libero/lb_data/lb_randsam_confs.yaml'
    # Open the YAML file and load its contents
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        rs_cfg = config[args.sub_conf]

    seed_u = rs_cfg['seed_u']
    random.seed(seed_u)
    np.random.seed(seed_u)
    torch.manual_seed(seed_u)


    el_name = rs_cfg['el_name']

    envlist: LiberoEnvList_V3
    envlist = gym.make(rs_cfg['el_name'],) #  gen_data=True)
    print(f'el_name: {el_name}')
    
    rand_explo_num_Ep_per_tk = rs_cfg['rand_explo_num_Ep_per_tk']

    all_ep_group_name = [] ## use for saving to h5
    all_ep_imgs = []
    all_ep_acts = []
    all_ep_ee_poses = []
    all_ep_env_seed = []

    for i_t, tk in enumerate(envlist.task_list):
        for i_ep in range(rand_explo_num_Ep_per_tk):
            env_idx = envlist.seed_sets[tk][0] ## this is dummy
            ## actual seed
            e_seed = np.random.randint(low=0, high=int(1e8))
            env = envlist.init_1_given_env(tk, env_idx=env_idx, e_seed=e_seed)
            
            ## -----------------
            ### do sampling here
            imgs_ep, acts_ep, ee_poses_ep = lb_rand_sample_1_ep(env_u=env, rs_cfg=rs_cfg)

            ## -----------------
            ## Finished, close env

            envlist.close_exist_env()
            ## Add to data
            all_ep_imgs.append(imgs_ep)
            all_ep_acts.append(acts_ep)
            all_ep_ee_poses.append(ee_poses_ep)

            all_ep_group_name.append(f'{tk}/{i_ep}/')
            all_ep_env_seed.append(e_seed)

            print(f'Current:{i_t} {i_ep}')


    assert len(all_ep_imgs) == envlist.num_tasks * rand_explo_num_Ep_per_tk


    h5_root = './data_dir/scratch/libero/env_rand_samples'
    os.makedirs(h5_root, exist_ok=True)
    h5_save_path = f'{h5_root}/{args.sub_conf}.hdf5'

    ## ----------------------------------
    ## Finished all, save to hdf5
    with h5py.File(h5_save_path, 'w') as file:
        # Directly create a nested group structure
        # group = file.create_group('Layer1/Layer2/Layer3/Layer4')
        for i_ep in range(len(all_ep_imgs)):
            grp_name = all_ep_group_name[i_ep]
            print('grp_name', grp_name)
            ## 1. create group
            if grp_name not in file:
                print(f"Creating group '{grp_name}'.")
                fgroup = file.create_group(grp_name)
            else:
                fgroup = file[grp_name]
            
            fgroup.attrs['env_seed'] = all_ep_env_seed[i_ep]
            fgroup.attrs['env_list_name'] = el_name

            ## dataset = group.create_dataset('Dataset1', data=[1, 2, 3, 4, 5])
            ## Now, add a dataset, i.e., storing data
            fgroup.create_dataset('agentview_image', data=all_ep_imgs[i_ep])
            fgroup.create_dataset('action', data=all_ep_acts[i_ep])
            fgroup.create_dataset('ee_poses', data=all_ep_ee_poses[i_ep])

    ## lock file
    if 'luotest' not in args.sub_conf:
        os.chmod(h5_save_path, 0o444)
    

if __name__ == '__main__':
    main()

