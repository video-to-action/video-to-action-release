import numpy as np
import torch, os, pdb
from typing import List
from environment.offline_env import OfflineEnv, PybulletEnv
import os.path as osp; from tqdm import tqdm; from einops import rearrange
import diffuser.utils as utils
LB_TASK_INTERACT_TYPE = {}
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv
from libero.libero import benchmark, get_libero_path, set_libero_default_path
from libero.libero.envs import OffScreenRenderEnv
from environment.libero.lb_utils import lb_full_cam_name


class LiberoEnvList_V3(OfflineEnv, PybulletEnv):

    def __init__(self,
                 task_suite_name: str, # "libero_90"
                 task_idx_list: List[int], 
                 num_envs_per_task: int, # 1
                 train_seed_start,
                 eval_seed_start,
                 dataset_dir,
                 dataset_name, # dir name of img dataset
                 envlist_cfg={},
                 np_seed=2727,
                 **kwargs):
        '''
        Wrap a list of metaworld envs to be trained on
        We instantiate N env per tasks
        '''

        self.task_suite_name = task_suite_name
        ## keys: 'libero_spatial', 'libero_object', 'libero_90', ...
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        self.task_idx_list = task_idx_list
        
        self.task_list = [] # list of str for model input
        self.task_dirname_list = []
        self.task_bddl_file_list = [] # a list of str, filepath
        self.task_to_task_idx = {} # tk_name to tk_idx
        self.task_idx_to_task = {} # tk_idx to tk_name
        for task_id in task_idx_list:
            # retrieve a specific task
            task = task_suite.get_task(task_id)
            task_name = task.name # 'LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate'
            self.task_dirname_list.append(task_name)
            
            ## input to the model
            task_description = task.language
            self.task_list.append(task_description) # 'put the red mug on the left plate'
            self.task_to_task_idx[task_description] = task_id
            self.task_idx_to_task[task_id] = task_description
            ## bddl file is used to init envs
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            self.task_bddl_file_list.append(task_bddl_file)
            utils.print_color(f"[info] retrieving task {task_id} from suite {task_suite_name}," 
                              f"the language instruction is {task_description}")

        ## ----------------------
        ## ----------------------
        ## ----------------------

        self.num_tasks = len(self.task_list)
        self.train_seed_start = train_seed_start
        self.num_envs_per_task = num_envs_per_task # 1
        self.n_total_envs = self.num_tasks * num_envs_per_task
        ## used to get obs
        self.actions_0_zero = np.zeros(shape=(self.n_total_envs, 7), dtype=np.float32)

        self.eval_seed_start = eval_seed_start # 0

        self.img_resol = envlist_cfg.get('img_resol', (128, 128)) # H, W


        self.kwargs = kwargs
        self.dataset_url = kwargs['dataset_url']
        self.env_name = self.get_env_name()
        self.camera_list = ['agent',] ## PlaceHolder, but cannot delete
        
        
        self.video_dir = osp.join(dataset_dir, dataset_name,)


        self.envlist_cfg = envlist_cfg



        OfflineEnv.__init__(self, **kwargs)
        dummy = np.zeros(shape=(7,), dtype=np.float32)
        PybulletEnv.__init__(self, obs_low=dummy, obs_high=dummy+1,
                            action_low=dummy,
                            action_high=dummy+1)


        ## -------- V2 NEW --------
        self.get_per_task_seed()
        self.np_seed = np_seed
        self.np_random = np.random.default_rng(self.np_seed)

        ### -------- UseLess
        self.env_act_scale = envlist_cfg['env_act_scale']
        self.env_fr_skip = envlist_cfg['env_fr_skip']
        self.env_mocap_minmax = envlist_cfg.get('env_mocap_minmax', None)
        self.env_act_repeat = envlist_cfg.get('env_act_repeat', None)

        self.ro_max_steps = envlist_cfg.get('ro_max_steps', 300)
        self.ro_extra_steps = envlist_cfg.get('ro_extra_steps', 10)

        self.task_interact_type = {**LB_TASK_INTERACT_TYPE}
        ## ------------------------


        self.init_env_list()
        self.dataset_name = dataset_name
        ## 
        self.num_seed_per_task = len( self.seed_sets[ self.task_list[0] ] )

        self.version = 'v3'
        # print(f'{self.num_seed_per_task}')
        
        # pdb.set_trace()
        
        ## ----------------------------
        ## ----------------------------


    ##  ------------------------------------------------    
    #### ------------ Initialization ------------------- 
    ##  ------------------------------------------------    
    def init_env_list(self):
        ''' 
        V2 NEW
        do it every time when init the class, 
        only create the env_args for init env
        '''
        self.env_init_states = {}
        self.env_list = {}
        self.actual_env_seeds = {}
        # self.all_envs_inst = []
        self.env_args_list = {}
        
        for i_t, tk_name in enumerate(self.task_list):
            self.env_list[tk_name] = {}
            self.actual_env_seeds[tk_name] = {}
            self.env_init_states[tk_name] = {}
            self.env_args_list[tk_name] = {}


            for i_s in self.seed_sets[tk_name]:
                ## create an env
                env_args = {
                    "bddl_file_name": self.task_bddl_file_list[i_t],
                    "camera_heights": self.img_resol[0], # 128
                    "camera_widths": self.img_resol[1], # 128
                    "camera_depths": True,
                    "horizon": 2000,
                }
                self.env_args_list[tk_name][i_s] = env_args
                
                

                
                self.env_list[tk_name][i_s] = None # env
                self.actual_env_seeds[tk_name][i_s] = i_s
                self.env_init_states[tk_name][i_s] = None

        





    def set_env_attr(self, env):

        return env
                
    def reset_env_list_all(self, is_rand):
        '''
        V2 NEW, Different from MW, only reset, not recreate!
        reset all envs in the env_list, to original or new random state
        '''
        assert is_rand

        from typing import Dict
        self.env_list: Dict[str, Dict[int, OffScreenRenderEnv]]

        for tk_name in self.task_list:

            for i_s in self.seed_sets[tk_name]:
                if is_rand:
                    tmp_s = self.np_random.integers(low=0, high=99999999).item()
                else:
                    ## reset to original
                    tmp_s = i_s
                

                self.actual_env_seeds[tk_name][i_s] = tmp_s
                self.env_init_states[tk_name][i_s] = None


    def init_1_given_env(self, tk_name, env_idx, e_seed=None, is_rand=True, with_seg=False):
        '''
        given e_seed, is_rand will be ignore
        '''
        ## init and sanity check
        self.check_no_envs_exist()
        env_args = self.env_args_list[tk_name][env_idx]
        if with_seg:
            env = SegmentationRenderEnv(**env_args)
        else:
            env = OffScreenRenderEnv(**env_args)

        env = self.set_env_attr(env)
        
        ## seeding
        if e_seed is not None:
            assert type(e_seed) == int
            env.seed(e_seed)
            tmp_s = e_seed
        else:
            if is_rand:
                tmp_s = self.np_random.integers(low=0, high=99999999).item()
            else:
                ## reset to original
                tmp_s = env_idx
            env.seed(tmp_s)
        
        # breakpoint()
        ## reset and step 0 to make objects in place
        obs = env.reset()
        
        self.env_list[tk_name][env_idx] = env
        self.actual_env_seeds[tk_name][env_idx] = tmp_s
        self.step_zero_act_1_env(tk_name, env_idx)

        
            
        
        
        return env

    
    def close_1_given_env(self, tk_name, env_idx,):
        '''close the env to prevent interfere'''

        env = self.env_list[tk_name][env_idx]
        env.close()
        del env
        self.env_list[tk_name][env_idx] = None
    
    def close_exist_env(self):
        '''should only be one env'''
        cnt_e = 0
        for tk_name in self.task_list:
            for i_s in self.seed_sets[tk_name]:
                env = self.env_list[tk_name][i_s]
                if env is not None:
                    env.close()
                    del env
                    self.env_list[tk_name][i_s] = None
                    cnt_e += 1
                    utils.print_color(f'found close_exist_env: {tk_name} {i_s}')
        assert cnt_e <= 1

    
    def check_no_envs_exist(self):
        '''should close all other before init new'''
        for tk_name in self.task_list:
            for i_s in self.seed_sets[tk_name]:
                assert self.env_list[tk_name][i_s] is None


    def reset_1_env(self, tk_name, env_idx, is_rand=True):
        '''
        May 11, just seed and reset
        '''
        # self.check_no_envs_exist()
        assert is_rand
        ## 'env_idx must be in the initial range'
        assert env_idx in self.env_list[tk_name].keys(), f'{env_idx}'
        if is_rand:
            tmp_s = self.np_random.integers(low=0, high=99999999).item()
        else:
            ## reset to original
            tmp_s = env_idx
        
        self.env_list[tk_name][env_idx] = \
            self.set_env_attr(self.env_list[tk_name][env_idx])
        self.env_list[tk_name][env_idx].seed(tmp_s)
        self.env_list[tk_name][env_idx].reset()

        self.actual_env_seeds[tk_name][env_idx] = tmp_s
        # self.env_init_states[tk][i_s] = self.env_list[tk][i_s].get_sim_state()

        self.step_zero_act_1_env(tk_name, env_idx)
        


    
   

    

    def step_zero_act_1_env(self, tk_name, env_idx,):
        '''

        '''
        # assert len(tasks_str) == len(env_idxs)
        # for i_sam, tk_name in enumerate(tasks_str):
            # i_s = env_idxs[i_sam]
            # env = self.env_list[tk_name][i_s]
        env = self.env_list[tk_name][env_idx]
        for _ in range(10):
            ret = env.step(self.actions_0_zero[0])
        return ret



    
    def get_per_task_seed(self):
        '''
        now actually we assume that the seeds for every task are the different.
        '''
        self.seed_sets = {} # a dict of set

        for i_tk, task_name in enumerate(self.task_list): # eg. put red mug ...
            tk_set = set()

            seed_start = self.train_seed_start + i_tk * self.num_envs_per_task
            seed_end = seed_start + self.num_envs_per_task

            for i_seed in range(seed_start, seed_end):
                tk_set.add( i_seed )

            ## transform the set to a sorted list, tested
            self.seed_sets[task_name] = sorted(tk_set)

        # utils.print_color( f'self.seed_sets: {self.seed_sets}' )
        utils.print_color( f'len self.seed_sets: {len(self.seed_sets)}' )

        return self.seed_sets
    
    
    
    ## ------------------------------------------------

    

    
    
    def render_an_env(self, task_name, cam_name, idx,):
        '''
        render corruption issues: 
            if concurrently init multiple libero inv, the render image might be corrupted.
        render return np 3d 128 128 3
        '''
        obs = self.env_list[task_name][idx].env._get_observations()
        # obs = self.env_list[task_name][idx].env._get_observations()

        cam_name = lb_full_cam_name(cam_name, is_depth=False)
        img = obs[cam_name]
        return img
    
    def render_an_env_with_preproc(self, task_name, cam_name, idx, imgs_preproc_fn):
        '''render one env img and followed by 
        imgs_preproc_fn: a batch level preprocessing func to convert np to a tensor
        (H,W,3) -> (3, H, W)
        '''
        img = self.render_an_env(task_name, cam_name, idx, )
        img = imgs_preproc_fn(img[None,])[0]
        
        assert img.ndim == 3 and img.shape[0] == 3
        assert torch.is_tensor(img)
        return img


    
    def render_an_env_with_depth(self, task_name, cam_name, idx):
        '''render return list of np1d
        V2 should have many envs operating concurrently, render twice
        '''

        obs = self.env_list[task_name][idx].env._get_observations()
        # obs = self.env_list[task_name][idx].env._get_observations()

        cam_1 = lb_full_cam_name(cam_name, is_depth=False)
        cam_dep = lb_full_cam_name(cam_name, is_depth=True)
        img = obs[cam_1]
        dep = obs[cam_dep]

        ## depth back to world meter 
        env = self.env_list[task_name][idx]
        extent = env.env.sim.model.stat.extent
        near = env.env.sim.model.vis.map.znear * extent
        far = env.env.sim.model.vis.map.zfar * extent
        ## NOTE: This is the real depth, no need of negative sign
        dep = near / (1 - dep * (1 - near / far))
        assert (dep >= 0).all(), 'sanity check'


        return img, dep
    
    @staticmethod
    def render_a_given_env(env: OffScreenRenderEnv, cam_name):
        obs = env.env._get_observations()
        cam_name = lb_full_cam_name(cam_name)
        img = obs[cam_name]
        return img
        
        
    

    @staticmethod
    def render_a_given_env_with_depth(env, cam_name,):
        

        obs = env.env._get_observations()

        cam_1 = lb_full_cam_name(cam_name, is_depth=False)
        cam_dep = lb_full_cam_name(cam_name, is_depth=True)
        img = obs[cam_1]
        dep = obs[cam_dep]

        extent = env.env.sim.model.stat.extent
        near = env.env.sim.model.vis.map.znear * extent
        far = env.env.sim.model.vis.map.zfar * extent
        ## NOTE: This is the real depth, no need of negative sign
        dep = near / (1 - dep * (1 - near / far))
        assert (dep >= 0).all(), 'sanity check'

        
        return img, dep

    
    
   
    
    def get_an_env_ref(self, task_name, idx):
        '''the return is only a reference'''
        return self.env_list[task_name][idx]
    

    def get_an_env_obs(self, task_name, idx) -> dict:
        '''the return is only a reference'''
        return self.env_list[task_name][idx].env._get_observations()
    
    def step_an_env(self, task_name, idx, act):
        ## obs, rew, done, info
        return self.env_list[task_name][idx].step(act)
    
    
    
    def get_random_env_ref(self):
        # just the first one, for check
        return next(iter(self.env_list[self.task_list[0]].values()))


    ## ------------------------------------------------

    
    ## --------------- Eval ----------------    
    
    
    
    def reset_env_instance(self, task_name, env_idx):
        self.env_list[task_name][env_idx].reset()
    

    def render_multi_imgs(self, tk, cams_list, env_idx, prev_dict=None):
        '''render multi cams imgs of one env'''
        result = {}
        for cam in cams_list:
            img = self.render_an_env(tk, cam, env_idx,)
            result[cam] = [img,]
            if prev_dict is not None: # dict of list
                prev_dict[cam].append(img)
        
        return result

    
    ## ----------------- utility ---------------------

    


    def get_env_name(self):
        hdf5_path = os.path.basename(self.dataset_url)
        env_name = hdf5_path.replace('.hdf5', '')
        utils.print_color(f'get_env_name: {env_name}')
        name = self.kwargs.get('name')
        if name:
            assert name == env_name
        return env_name
    

    ## --------------------------------
    




    def unload_all_envs(self):
        '''release all resources'''
        for i_t, tk_name in enumerate(self.task_list):
            for i_s in self.seed_sets[tk_name]:
                env = self.env_list[tk_name][i_s] 
                if env is not None and hasattr(env, 'env'):
                    env.close()
    
    def __del__(self):
        self.unload_all_envs()

    def __exit__(self):
        self.unload_all_envs()



    


