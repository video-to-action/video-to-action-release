import os
os.environ['MUJOCO_GL'] = 'egl' ## gpu
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"
from gym.envs.registration import register

root_dir = '/oscar/data/csun45/yluo73/r2024/AVDC_experiments/data_dir/env_cfg/'
LB_data_dir = '/oscar/data/csun45/yluo73/r2024/LIBERO/libero/datasets_img/'

LB_8tk_65_72 = [65,66,67,68,69,70,71,72]
LB_1tk_71_x8 = [71,] * 8
LB_1tk_71_x1 = [71,]
LB_1tk_69_x1 = [69,]
Dset_LB_8tk_65_72 = 'libero_90_65To72' ## data dir title

def register_libero():

    ts_st = 10000
    register(
        id='libero-8tk-65to72-v3',
        entry_point='environment.libero.lb_env_v3:LiberoEnvList_V3',
        max_episode_steps=250,
        kwargs={
            'task_suite_name': 'libero_90',
            'task_idx_list': LB_8tk_65_72,
            'num_envs_per_task': 1,
            
            'train_seed_start': ts_st,
            'envlist_cfg': dict(
                env_act_scale=None,
                env_fr_skip=None,
                env_mocap_minmax=None,
                env_act_repeat=None,
                ),
            
            'eval_seed_start': 100,
            'dataset_dir': LB_data_dir,
            'dataset_name': Dset_LB_8tk_65_72,
            'dataset_url':f'{root_dir}/libero-8tk-65to72-v3.hdf5',
        }
    )

    

    
    ## ----------------------------------------------
    ## ---------------- LuoTest ---------------------

    ts_st = 10000
    register(
        id='libero-2tk-65n72-luotest-v3',
        entry_point='environment.libero.lb_env_v3:LiberoEnvList_V3',
        max_episode_steps=250,
        kwargs={
            'task_suite_name': 'libero_90',
            'task_idx_list': [65, 72,],
            'num_envs_per_task': 1,
            
            'train_seed_start': ts_st,
            'envlist_cfg': dict(
                env_act_scale=None,
                env_fr_skip=None,
                env_mocap_minmax=None,
                env_act_repeat=None,
                ),
            
            'eval_seed_start': 0,
            'dataset_dir': LB_data_dir,
            'dataset_name': Dset_LB_8tk_65_72,
            'dataset_url':f'{root_dir}/libero-2tk-65n72-luotest-v3.hdf5',
        }
    )



    ts_st = 10000
    register(
        id='libero-2tk-65n72-luotest-v2',
        entry_point='environment.libero.lb_env_v2:LiberoEnvList_V2',
        max_episode_steps=250,
        kwargs={
            'task_suite_name': 'libero_90',
            'task_idx_list': [65, 72,],
            'num_envs_per_task': 1,
            
            'train_seed_start': ts_st,
            'envlist_cfg': dict(
                env_act_scale=None,
                env_fr_skip=None,
                env_mocap_minmax=None,
                env_act_repeat=None,
                ),
            
            'eval_seed_start': 0,
            'dataset_dir': LB_data_dir,
            'dataset_name': Dset_LB_8tk_65_72,
            'dataset_url':f'{root_dir}/libero-2tk-65n72-luotest-v2.hdf5',
        }
    )


    ts_st = 10000
    register(
        id='libero-2tk-65-66-luotest-v0',
        entry_point='environment.libero.lb_env:LiberoEnvList',
        max_episode_steps=250,
        kwargs={
            'task_suite_name': 'libero_90',
            'task_idx_list': [65, 66,],
            'num_envs_per_task': 1,
            
            'train_seed_start': ts_st,
            'envlist_cfg': dict(
                env_act_scale=None,
                env_fr_skip=None,
                env_mocap_minmax=None,
                env_act_repeat=None,
                ),
            
            'eval_seed_start': 0,
            'dataset_dir': LB_data_dir,
            'dataset_name': Dset_LB_8tk_65_72,
            'dataset_url':f'{root_dir}/libero-2tk-65-66-luotest-v0.hdf5',
        }
    )


