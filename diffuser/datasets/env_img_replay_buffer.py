from collections import deque
import random; from collections import namedtuple
import numpy as np; import torch, pdb
import diffuser.utils as utils
from einops import rearrange; from typing import List
from diffuser.datasets.img_utils import save_img_tr, save_gif_tr
from environment.libero.lb_env_v3 import LiberoEnvList_V3
from itertools import islice

class Global_EnvReplayBuffer_Img:
    
    def __init__(self, task_list, 
                 max_num_unitBufs,
                 max_len_uB,
                 min_len_uB,
                 env_list: LiberoEnvList_V3, render_img_size, env_buf_config={}):
        self.buffers: deque[EnvImg_UnitBuffer] = deque(maxlen=max_num_unitBufs)
        
        self.task_list = task_list
        
        
        self.env_list = env_list # same list as in the trainer
        ## for randomly sampled camera angle
        self.camera_list = self.env_list.camera_list
        self.bufs_task = deque(maxlen=max_num_unitBufs)
        self.bufs_cam = deque(maxlen=max_num_unitBufs) 

        self.num_cams = len(self.camera_list)
        self.render_img_size = render_img_size
        
        self.sample_act_seq_len = env_buf_config['sample_act_seq_len'] # for sampling
        

        # assert self.buf_full_len < 1e4
        self.per_sample_gap = 1 # should be useless for now
        self.max_num_unitBuf = max_num_unitBufs
        self.max_len_uB = max_len_uB
        self.min_len_uB = min_len_uB
        ## how many episodes has been added (including those removed one)
        ## can be used as criterion to stop env access for sample efficiency
        self.cnt_all_history_episodes = 0
        assert max_num_unitBufs <= 1e4
    
        
    def add_one_episode(self, tk: str, cam_name, env_idx, 
                        imgs: List[torch.Tensor], acts: List[torch.Tensor], is_suc=False):
        '''add a whole env episode rollout, env is supposed to be reset after that'''
        ## -------
        assert len(self.buffers) <= self.max_num_unitBuf
        assert len(imgs) == len(acts) + 1
        ## init
        tmp_buf = EnvImg_UnitBuffer(self.max_len_uB, tk, 
                                    cam_name=cam_name, env_idx=env_idx, per_sample_gap=self.per_sample_gap)
        tmp_buf.push_seq(new_imgs=imgs, new_acts=acts)
        
        assert self.min_len_uB <= len(tmp_buf)
        
        ## append multiple info
        self.buffers.append(tmp_buf)
        self.bufs_task.append(tk)
        self.bufs_cam.append(cam_name)

        self.cnt_all_history_episodes += 1




    def sample_random_batch_seq(self, batch_size,):
        """randomly samples a batch of experiences (act len > 1) from the buffer.
        len of each data sample == self.sample_hzn
        Return:
        imgs_start, imgs_goal: cpu tensor (B C H W)
        acts: cpu tensor (B, act_len, 4)
        tasks_str: list of str
        """
        # cur_len = len(self.buf_keys_list)
        assert len(self.buffers) == len(self.bufs_task) == len(self.bufs_cam)
        cur_len = len(self.buffers)
        
        ## number can repeat, sample 'task+env' together
        buf_selected_idxs = np.random.randint(0, cur_len, size=batch_size)

        

        imgs_start, imgs_goal = [], [] # should be a list cpu tensor
        acts_seq, tasks_str, env_idxs = [], [], [] # should have '-'
        cams_str = []
        
        for bf_idx in buf_selected_idxs:
            ## return list of img_st, img_goal, tk_name
            # img_st, img_g, act, tk_name, env_idxs 
            # dp = self.buffer[self.buf_keys_list[bk_idx]].sample_seq(self.sample_act_seq_len)
            dp = self.buffers[ bf_idx ].sample_seq(self.sample_act_seq_len)
            
            imgs_start.append( dp[0] )
            imgs_goal.append( dp[1] )
            acts_seq.append( dp[2] )
            tasks_str.append( dp[3] )
            env_idxs.append( dp[4] )
            cams_str.append(self.buffers[ bf_idx ].cam_name)
        
        
        # imgs_start: list of (3,128,128)
        imgs_start = torch.stack(imgs_start, dim=0) # B,3,128,128
        imgs_goal = torch.stack(imgs_goal, dim=0)

        ## acts_seq is a list of tensor, after (B, self.act_seq_len, 4)
        acts_seq = torch.stack( acts_seq, dim=0 )
        tmp_info = dict(env_idxs=np.array(env_idxs), 
                        cams_str=cams_str,)
        

        assert imgs_start.ndim == 4 and acts_seq.ndim == 3
        assert not imgs_goal.is_cuda and not acts_seq.is_cuda
        
        return imgs_start, imgs_goal, acts_seq, tasks_str, tmp_info
    



    #### -------------- For the Ablation, One Action ResNet Model ---------------
    def sample_random_batch_seq_single_action(self, batch_size, max_future_hzn: int):
        """
        randomly samples a batch of experiences from the buffer.
        s1  s2  s3 ,..., s_{rand}, ..., s_{max}
          a1
        only returns s1, a1, s_rand
        Return:
        imgs_start, imgs_goal: cpu tensor (B C H W)
        acts: cpu tensor (B, 1, 4), ## we only need the current action, just like BC
        tasks_str: list of str
        """
        # cur_len = len(self.buf_keys_list)
        assert len(self.buffers) == len(self.bufs_task) == len(self.bufs_cam)
        cur_len = len(self.buffers)
        
        ## number can repeat, sample 'task+env' together
        buf_selected_idxs = np.random.randint(0, cur_len, size=batch_size)

        imgs_start, imgs_goal = [], [] # should be a list cpu tensor
        acts_seq, tasks_str, env_idxs = [], [], [] # should have '-'
        cams_str = []
        
        assert max_future_hzn <= 30, 'might use 16 to align with ours'
        
        for bf_idx in buf_selected_idxs:

            # dp = self.buffers[ bf_idx ].sample_seq(self.sample_act_seq_len)

            rand_f_hzn = random.randint(1, max_future_hzn)
            dp = self.buffers[ bf_idx ].sample_seq(rand_f_hzn)

            
            imgs_start.append( dp[0] )
            imgs_goal.append( dp[1] )

            # acts_seq.append( dp[2] )
            ## we only take the first action
            acts_seq.append( dp[2][0:1] ) # tensor e.g., [1, 7]


            tasks_str.append( dp[3] )
            env_idxs.append( dp[4] )
            cams_str.append(self.buffers[ bf_idx ].cam_name)
        
        
        # imgs_start: list of (3,128,128)
        imgs_start = torch.stack(imgs_start, dim=0) # B,3,128,128
        imgs_goal = torch.stack(imgs_goal, dim=0)

        ## acts_seq is a list of tensor, after (B, 1, 7)

        acts_seq = torch.stack( acts_seq, dim=0 )
        tmp_info = dict(env_idxs=np.array(env_idxs), 
                        cams_str=cams_str,)
        
        # pdb.set_trace()
        assert imgs_start.ndim == 4 and acts_seq.ndim == 3
        assert not imgs_goal.is_cuda and not acts_seq.is_cuda
        
        # pdb.set_trace() # acts_seq: (B, 1, 7)
        
        return imgs_start, imgs_goal, acts_seq, tasks_str, tmp_info













    
    
    

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffers)
    
    def __getitem__(self, idx):
        if idx >= len(self.buffers):
            raise IndexError("index out of range")
        return self.buffers[idx]

    
    def is_full(self):
        '''if the buffer is full'''
        return len(self.buffers) == self.max_num_unitBuf


## -----------------------------------------------------
## ---------- Unit Buffer for Env Interation -----------

class EnvImg_UnitBuffer:

    def __init__(self, max_len, task_name, cam_name, env_idx, 
                 per_sample_gap=1, max_start_id=None):
        '''
        This class only support one image observation (cam view).
        store a sequence of imgs of one specific env,
        make sure to terminate it once the env is reset.
        per_sample_gap: the gap between each (or two) env samples (not used for now)
        '''
        self.max_len = max_len # max history len

        self.per_sample_gap = per_sample_gap
        assert self.per_sample_gap == 1
        self.task_name = task_name
        self.cam_name = cam_name
        self.env_idx = env_idx

        assert type(task_name) == str
        # assert '-' not in task_name
        self.env_idx = env_idx
        self.imgs_buf: deque[torch.Tensor] = deque(maxlen=self.max_len) # list of np3d, a sequence of imgs
        self.acts = deque(maxlen=self.max_len-1) # action transition
    
    
        

    
    


    def push_seq(self, new_imgs: List[torch.Tensor], new_acts: List[torch.Tensor]):
        '''
        Direct push a list of imgs/actions to the buffer.
        We assume the input states include the last img in the buffer, 
        which is also the starting state of the input actions
        '''
        assert type(new_imgs) == list and type(new_acts) == list
        assert torch.is_tensor(new_imgs[0])
        assert not new_acts[0].is_cuda and not new_imgs[0].is_cuda
        assert len(new_acts) == len(new_imgs) - 1
        assert 2 <= len(new_imgs) <= 800, 'can be large when video rollout'

        if len(self.imgs_buf) == 0:
            self.imgs_buf.extend(new_imgs)
            self.acts.extend(new_acts)
        else:
            # pdb.set_trace()
            # assert (self.states[-1][0].qpos == new_states[0][0].qpos).all()
            assert self.imgs_buf[-1].shape == new_imgs[0].shape
            test_tmp = torch.sum( torch.abs(self.imgs_buf[-1] - new_imgs[0]) > 1e-3 )
            utils.print_color(f'Buf num diff pixels: {test_tmp}')

            self.imgs_buf.extend( new_imgs[1:] )
            self.acts.extend( new_acts )

        # pdb.set_trace()
        assert len(self.imgs_buf) <= self.max_len
    
    def sample_seq(self, act_seq_len):
        """
        sample a list of consecutive images and actions, used when the sim frameskip is tiny
        Returns:
        img_start, img_goal: two single states
        ret_acts: tensor of actions (act_seq_len)
        s1   s2    s3    s4
           a1   a2    a3
        """
        cur_len = len(self.imgs_buf)
        assert act_seq_len < cur_len
        start_idx = random.randint(0, cur_len - act_seq_len - 1 ) # [a,b] inclusive, checked

        goal_idx = start_idx + act_seq_len # this idx is not included when sample action

        # [No, delete] ret_states: list of states (act_seq_len+1)
        # ret_states = list( islice(self.states, start_idx, goal_idx) ) # [a,) exclusive

        ret_acts = list( islice( self.acts, start_idx, goal_idx ) ) # list of tensor
        # pdb.set_trace()
        ret_acts = torch.stack( ret_acts, dim=0 ) # (len,4)
        
        assert len(ret_acts) == act_seq_len

        return self.imgs_buf[start_idx], self.imgs_buf[goal_idx], ret_acts, self.task_name, self.env_idx




    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.imgs_buf)
