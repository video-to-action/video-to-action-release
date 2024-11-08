from pathlib import Path
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import sys, os, torch, pdb, imageio, random, h5py
from diffuser.utils.luo_utils import batch_repeat_tensor, get_lr
import diffuser.utils as utils
from torchvision import transforms as T
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator
import matplotlib.pyplot as plt; import numpy as np
from flowdiffusion.flowdiffusion.goal_diffusion import cycle, exists, print_gpu_utilization, GoalGaussianDiffusion
from environment.libero.lb_env_v3 import LiberoEnvList_V3
from diffuser.datasets.img_utils import save_img_tr, save_gif_tr, imgs_preproc_simple_noCrop_v1
from diffuser.datasets.env_img_replay_buffer import Global_EnvReplayBuffer_Img
from diffuser.models.train_utils import freeze_model, freeze_trainer, merge_batch, weighted_uniform_sample, identity_tensor
from diffuser.models.video_model import Video_PredModel
import os.path as osp
import wandb; from copy import deepcopy; __version__ = 'Nov-2024-release'
from diffuser.utils.rendering import MW_Renderer
from diffuser.utils import plot2img
from functools import partial
from diffuser.diffusion_policy import Init_Diffusion_Policy, DiffusionUnetImagePolicy

class LB_Online_Trainer_V7(object):
    def __init__(
        self,
        init_diff_policy: Init_Diffusion_Policy,
        video_model,
        tokenizer, 
        text_encoder,
        train_set,
        valid_set,
        channels = 3,
        *,
        train_batch_size = 1, ## random explore bs
        video_batch_size = 1,
        valid_batch_size = 1,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_num_steps = 100000,
        opt_params,
        ema_params,
        
        render_img_size=(320, 240),
        input_img_size=(128, 128),

        sample_freq=1000,
        save_freq=1000,
        label_freq=50000,
        log_freq=100,

        num_samples = 3,
        results_folder = './results',
        amp = True,
        fp16 = True,
        split_batches = False, # True,

        trainer_dict,
        **kwargs,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        # accelerator
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp

        # model
        self.gcp_model: DiffusionUnetImagePolicy = init_diff_policy.diffusion_policy
        ## a nn.Module
        self.video_model: Video_PredModel = self.accelerator.prepare(video_model)

        assert video_model.ema.ema_model.image_size == input_img_size
        
        freeze_model(self.video_model)

        self.channels = channels ## ?? input channels?

        # sampling and training hyperparameters

        self.num_samples = num_samples
        self.sample_freq = sample_freq # eval freq
        self.save_freq = save_freq # freq to save the checkpoint
        self.label_freq = label_freq # how many checkpoint to save
        self.render_img_size = render_img_size # size rendered 
        self.input_img_size = input_img_size # size input to the model
        self.log_freq = log_freq


        self.batch_size_rand = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        

        self.env_list: LiberoEnvList_V3 = train_set.env_list

        self.task_list = self.env_list.task_list # a list of str
        

        ## dataset and dataloader
        self.train_set = train_set
        ## number of task, cam, env combinations
        self.train_set_len = len(train_set) ## 1000
        

        # create dataloader for random exploration
        dl = DataLoader(self.train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=0)
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        
        # create dataloader for guided exploration
        assert video_batch_size == 1
        dl_vid = DataLoader(self.train_set, batch_size=video_batch_size, shuffle=True, pin_memory=True, num_workers=0)
        self.dl_vid = self.accelerator.prepare(dl_vid)
        

        # optimizer
        self.opt = AdamW(self.gcp_model.parameters(), **opt_params)
        self.opt.zero_grad()
        
        # create EMA model
        if self.accelerator.is_main_process:
            self.ema = EMA(self.gcp_model, **ema_params)
            self.ema.to(self.device) # will move both models
            self.ema.ema_model.normalizer.to_device(self.device)

        self.gcp_model.to(self.device)
        self.text_encoder.to(self.device)
        assert not self.text_encoder.training

        # create save folder
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = 0
        self.num_steps_in_env = 0
        
        # pdb.set_trace()
        # prepare model, optimizer with accelerator
        self.gcp_model, self.opt = \
            self.accelerator.prepare(self.gcp_model, self.opt)
        
        ## Used in Random sample actions
        ## for clamp and rand sample
        self.act_min_np, self.act_max_np = self.train_set.act_min_max
        assert (np.abs(self.act_min_np) == self.act_max_np).all(), 'necessary for our sampling'
        self.act_min, self.act_max = \
                torch.tensor(self.act_min_np), torch.tensor(self.act_max_np)
        self.act_dim = len(self.act_max)

        
        self.trainer_dict = trainer_dict
        self.lr_warmupDecay = trainer_dict.get('lr_warmupDecay', False)
        ## either train or eval, this is set to 'eval' inside load_lb_diffusion
        self.cur_mode = trainer_dict.get('cur_mode', 'train')

        # create buffer
        self.init_helpers()
        self.loss_fn = nn.MSELoss()
        self.control_mode = 'delta'
        # pdb.set_trace()


    def init_helpers(self):
        '''
        task_env_states: store the lastest state of each seed; num_tasks * num_seed
        '''


        self.num_envs_per_tk = self.env_list.num_seed_per_task
        
        assert self.num_envs_per_tk == 1 ## or self.trainer_dict['save_to_scratch']
        self.num_envs_all_tks = self.num_envs_per_tk * self.env_list.num_tasks
        assert self.train_set_len == self.num_envs_all_tks
        utils.print_color(f'self.num_envs_all_tks: {self.num_envs_all_tks}')
        
        ## ------ Env Interation Buffer --------
        

        self.envBuf_max_num_uB_rand = self.trainer_dict['envBuf_max_num_uB_rand']
        self.envBuf_max_num_uB_vid = self.trainer_dict['envBuf_max_num_uB_vid']
        self.max_len_uB = self.trainer_dict['max_len_uB']
        self.min_len_uB = self.trainer_dict['min_len_uB']
        
        utils.print_color(f'envBuf maxlen:rand:{self.envBuf_max_num_uB_rand},vid:{self.envBuf_max_num_uB_vid}, '
                          f'uB: max{self.max_len_uB}, min{self.min_len_uB}')

        assert self.envBuf_max_num_uB_rand >= 1000
        assert self.envBuf_max_num_uB_vid >= 500 ## or self.trainer_dict['save_to_scratch']

        self.model_act_horizon = self.trainer_dict['model_act_horizon'] # 24, used in sampling        

       

        ## ---------- NEW Init Image Buf, Apr 29 --------------
        self.envBuf_rand: Global_EnvReplayBuffer_Img
        self.envBuf_rand = self.get_new_env_buffer(self.envBuf_max_num_uB_rand)

        self.envBuf_vid: Global_EnvReplayBuffer_Img
        self.envBuf_vid = self.get_new_env_buffer(self.envBuf_max_num_uB_vid)

        # self.num_init_rand_episodes = self.trainer_dict['num_init_rand_Ep'] # e.g., 100
        self.num_init_rand_episodes_per_tk = self.trainer_dict['num_init_rand_Ep_per_tk'] # e.g., 10
        # assert self.num_init_rand_episodes_per_tk <= 200


        # self.num_env_access_per_rand_episodes = self.trainer_dict['num_env_access_per_rand_Ep'] # e.g., 10
        
        self.buf_sample_method = self.trainer_dict.get('buf_sample_method', 'iter_bias_fix')
        assert self.buf_sample_method in ['iter_bias_fix', 'rand_prob', 'iter_bias_rand']
        if self.buf_sample_method == 'rand_prob':
            self.buf_sample_randBuf_prob = self.trainer_dict['buf_sample_randBuf_prob']

        ## ---------------------------------------

        rs_fname = self.trainer_dict['randsam_filename']
        tmp_p1 = './data_dir/scratch/libero/env_rand_samples/'
        self.randsam_file_path = osp.join(tmp_p1, rs_fname)
        is_h5_exist = os.path.exists(self.randsam_file_path)
        assert self.cur_mode in ['train', 'eval']
        if not is_h5_exist:
            if self.cur_mode == 'eval':
                ## use a placeholder if do evaluation but no downloading the random action data
                self.randsam_file_path = osp.join(tmp_p1, 'lb_randsam_8tk_dummy_example.hdf5')
            else:
                assert False, 'prepare the random action dataset before training'

        ## get how many ep per task in the h5
        with h5py.File(self.randsam_file_path, 'r') as rs_file:
            for i_t, tk in enumerate(self.task_list):
                fgroup = rs_file[f'{tk}']
                f_keys = sorted([int(k) for k in fgroup.keys()])
                break
        self.h5_total_num_ep_per_task = f_keys[-1] + 1 ## e.g., 200

        if is_h5_exist and self.cur_mode == 'train':
            assert self.h5_total_num_ep_per_task >= self.num_init_rand_episodes_per_tk


        ## to store states of all envs
        ## copy initial states
        self.task_env_states: dict = deepcopy(self.env_list.env_init_states)

        
        
        self.rendered_imgs_preproc_fn = imgs_preproc_simple_noCrop_v1
        self.renderer = MW_Renderer(self.env_list)

        ## ----- Iteration Logic Control -----
        self.iter_type = 'rand-bias' # start from rand exploration
        self.init_rand_steps = self.trainer_dict['init_rand_steps']
        self.rand_cycle_steps = self.trainer_dict['rand_cycle_steps']
        self.vid_cycle_steps = self.trainer_dict['vid_cycle_steps']


        ## reset freq when doing random exploration
        # self.rand_iter_reset_freq = self.trainer_dict['rand_iter_reset_freq']
        self.use_env_rand_reset = self.trainer_dict['use_env_rand_reset']

        self.video_explo_freq = self.trainer_dict['video_explo_freq']
        self.rand_explo_freq = self.trainer_dict['rand_explo_freq']
        self.rand_explo_type = self.trainer_dict['rand_explo_type']
        assert self.rand_explo_type in ['from_env', 'from_h5']
        assert self.rand_explo_type == 'from_h5'

        # self.rand_explo_num_episodes = self.trainer_dict['rand_explo_num_Ep']
        self.rand_explo_num_episodes_per_tk = self.trainer_dict['rand_explo_num_Ep_per_tk']

        ### --- New
        self.explo_type_rand = 'explo' # start from rand exploration
        self.explo_type_vid = 'explo' # start from rand exploration
        
        self.enable_noExp = self.trainer_dict.get('enable_noExp', False)
        self.noExp_start_buf_len_rand = self.trainer_dict.get('noExp_start_buf_len_rand')
        self.noExp_start_buf_len_vid = self.trainer_dict.get('noExp_start_buf_len_vid')
        self.Exp_noExp_rand = self.trainer_dict.get('Exp_noExp_rand') # (1000, 2000)
        self.Exp_noExp_vid = self.trainer_dict.get('Exp_noExp_vid') # (1000, 2000)
        
        self.cnt_no_exp_rand = 0
        self.cnt_exp_rand = 0
        self.cnt_no_exp_vid = 0
        self.cnt_exp_vid = 0
        

        # assert self.rand_cycle_steps % self.rand_iter_reset_freq == 0
        if self.init_rand_steps != -1: # -1 is a special case
            assert self.init_rand_steps % self.rand_cycle_steps == 0

        self.rand_iter_cnt = 0
        self.vid_iter_cnt = 0
        
        ## ------- Guided Rollout ---------
        ## num of success in the train time exploration
        self.cnt_explore_suc = 0
        self.cnt_vid_rollouts = 0
        
        self.cnt_explo_suc_per_tk = {tk: 0 for tk in self.task_list }
        self.cnt_vid_rout_per_tk = {tk: 0 for tk in self.task_list }

        ## execute the first 8 acts out of model_act_horizon(24) acts
        self.n_acts_per_pred = self.trainer_dict['n_acts_per_pred'] # 8
        self.n_preds_betw_vframes = self.trainer_dict['n_preds_betw_vframes'] # tuple,range, e.g.(5,6)
        # assert self.rand_act_full_len >= self.model_act_horizon, 'must be longer to sample'
        ## can be smaller
        utils.print_color(f'self.n_preds_betw_vframes: {self.n_preds_betw_vframes }')
        
        self.max_acts_betw_vframes = self.n_acts_per_pred * self.n_preds_betw_vframes[1] # e.g. 40,
        ## total actions in one rollout, e.g. 7*5*8=280
        total_rollout_acts = self.max_acts_betw_vframes * self.video_model.video_future_horizon
        assert self.max_len_uB > total_rollout_acts, 'must store all rollout'


        ## ------ Loss computation --------
        self.buf_sample_batch_size = self.trainer_dict['buf_sample_batch_size']

        ## we use a different ratio to empahsize differently, trade-off rand and guided samples
        # [rand, guided], e.g., [0.5, 0.5]
        bsr_rand = self.trainer_dict['buf_sample_ratio_rand']
        bsr_vid = self.trainer_dict['buf_sample_ratio_vid']
        # how many samples from each buffer, e.g., [8, 8]
        self.nums_buf_sample_rand = utils.number_by_ratio(self.buf_sample_batch_size, bsr_rand)
        self.nums_buf_sample_vid_bias = utils.number_by_ratio(self.buf_sample_batch_size, bsr_vid)
        utils.print_color(f'ratio rand:{self.nums_buf_sample_rand}, vid: {self.nums_buf_sample_vid_bias}')
        self.r_prob_rand = bsr_rand[0] # probability of a random sample when rand-bias
        self.r_prob_vid = bsr_vid[0] # probability of a random sample when video-bias

        
        self.n_acts_down_range = self.trainer_dict['n_acts_down_range'] # 16
        self.n_acts_close_grp = self.trainer_dict['n_acts_close_grp'] # 8
        self.act_down_val = self.trainer_dict['act_down_val'] # -0.98

        if 'act_down_val_range_per_tk' in self.trainer_dict:
            self.act_down_val_range_per_tk = self.trainer_dict['act_down_val_range_per_tk']
            assert self.act_down_val is None
        else:
            ## just use normal all same down val
            assert self.act_down_val <= -0.5

        self.close_grp_force = self.trainer_dict['close_grp_force'] # 0.98
        self.close_grp_act_down_val = self.trainer_dict['close_grp_act_down_val']
        assert self.close_grp_act_down_val <= 0

        self.is_all_randsam_visited = False ## not perfectly match

        self.is_stop_at_suc = self.trainer_dict['is_stop_at_suc']




    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'num_steps_in_env': self.num_steps_in_env,
            'gcp_model': self.accelerator.get_state_dict(self.gcp_model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__,
            ##
            'cnt_vid_rollouts': self.cnt_vid_rollouts,
            'cnt_vid_rout_per_tk': self.cnt_vid_rout_per_tk,
        }
        savepath = str(self.results_folder / f'model-{milestone}.pt')
        torch.save(data, savepath)
        utils.print_color(f'[ utils/training ] Saved model to {savepath}', c='y')


    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        gcp_model = self.accelerator.unwrap_model(self.gcp_model)
        gcp_model.load_state_dict(data['gcp_model'])

        self.step = data['step']
        self.num_steps_in_env = data['num_steps_in_env']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])



    def encode_batch_text(self, batch_text):
        batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.device)
        ## torch.Size([1, 4=n_words+2, 512])
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state
        return batch_text_embed
    
    
    def reset_all_envs(self):
        
        self.env_list.check_no_envs_exist()
    

    def reset_given_envs(self, tasks_str, env_idxs):
        
        self.env_list.recreate_given_envs(tasks_str, env_idxs, is_rand=True)
        for i_sam, tk in enumerate(tasks_str):
            e_idx = env_idxs[i_sam]
            assert e_idx in self.task_env_states[tk].keys()
            self.task_env_states[tk][e_idx] = self.env_list.env_init_states[tk][e_idx]

    
    def update_explo_type(self,):
        if not self.enable_noExp:
            return

        if len(self.envBuf_rand) >= self.noExp_start_buf_len_rand:
            if self.explo_type_rand == 'no-explo':
                self.cnt_no_exp_rand += 1
            elif self.explo_type_rand == 'explo':
                self.cnt_exp_rand += 1
            else:
                assert False

        if self.cnt_exp_rand == self.Exp_noExp_rand[0]:
            self.cnt_exp_rand = 0
            self.explo_type_rand = 'no-explo'

        if self.cnt_no_exp_rand == self.Exp_noExp_rand[1]:
            self.cnt_no_exp_rand = 0
            self.explo_type_rand = 'explo'

        ## --------
        if len(self.envBuf_vid) >= self.noExp_start_buf_len_vid:
            if self.explo_type_vid == 'no-explo':
                self.cnt_no_exp_vid += 1
            elif self.explo_type_vid == 'explo':
                self.cnt_exp_vid += 1
            else:
                assert False

            if self.cnt_exp_vid == self.Exp_noExp_vid[0]:
                self.cnt_exp_vid = 0
                self.explo_type_vid = 'no-explo'

            if self.cnt_no_exp_vid == self.Exp_noExp_vid[1]:
                self.cnt_no_exp_vid = 0
                self.explo_type_vid = 'explo'
            

        

    def train(self):
        self.debug = True # True
        timer = utils.Timer()
        accelerator = self.accelerator
        device = accelerator.device
        self.gcp_model.train()
        self.init_wandb_metrics()
        wandb.define_metric("explo/cnt_vid_rollouts")
        wandb.define_metric("train/cnt_explore_suc_vsR", step_metric="explo/cnt_vid_rollouts")

        ## Fill the random envBuf first
        print('Start training, fill init rand buf')
        if True:
            self.h5_add_rand_act_episodes_to_Buf(0, self.num_init_rand_episodes_per_tk)
            assert len(self.envBuf_rand) == \
                self.env_list.num_tasks * self.num_init_rand_episodes_per_tk
            ## the starting h5 ep idx of the next adding to envBuf
            self.h5_randsam_start_idx = self.num_init_rand_episodes_per_tk


        while self.step < self.train_num_steps:

            total_loss = 0.

            for _ in range(self.gradient_accumulate_every):

                self.update_iter_type()
                self.update_explo_type()
                ## before the start of noExplore, fill the env buffer vid
                # if self.iter_type == 'noExplore' and self.noExplore_iter_cnt == 0:

                ## update (e.g. 12) video rollout to the env vid buf every X train steps
                if self.step > self.init_rand_steps and \
                    self.step % self.video_explo_freq == 0 and \
                    self.explo_type_vid == 'explo':
                    self.video_guided_explore()
                    # self.envBuf_rand_need_reset = True
                
                ## add more random episodes to rand buf
                if self.step > self.init_rand_steps and \
                    self.step % self.rand_explo_freq == 0 and \
                    self.explo_type_rand == 'explo':
                    if self.rand_explo_type == 'from_env':
                        raise NotImplementedError
                        self.add_rand_act_episodes_to_Buf(self.rand_explo_num_episodes_per_tk)
                    elif self.rand_explo_type == 'from_h5':
                        ## this cirular design not ideal, but should work
                        tmp_st_idx = self.h5_randsam_start_idx % self.h5_total_num_ep_per_task
                        tmp_num_added_ep = min(self.h5_total_num_ep_per_task - tmp_st_idx, 
                            self.rand_explo_num_episodes_per_tk)
                        tmp_end_idx = tmp_st_idx + tmp_num_added_ep
                        self.h5_add_rand_act_episodes_to_Buf(tmp_st_idx, tmp_end_idx)
                        self.h5_randsam_start_idx += tmp_num_added_ep
                        if self.h5_randsam_start_idx >= self.h5_total_num_ep_per_task:
                            self.is_all_randsam_visited = True
                
                if self.iter_type == 'rand-bias':
                    self.rand_iter_cnt += 1

                
                elif self.iter_type == 'vid-bias':
                    assert len(self.envBuf_rand) > 0 or self.init_rand_steps == -1
                    # assert len(self.envBuf_rand_deque[-1]) > 10
                    self.vid_iter_cnt += 1
                else:
                    raise NotImplementedError()
                 

                ## LUOTEST
                if self.debug and len(self.envBuf_rand) > 0:
                    tmp_uB = self.envBuf_rand[-1]
                    if self.step % 500 == 0: # and self.step > 0:, % 30
                        img_r = self.renderer.render_img_grid(list(tmp_uB.imgs_buf)[-30:])
                        save_img_tr(img_r, self.results_folder, 'render_imgs_2', 
                                    f'{tmp_uB.task_name}', f'{tmp_uB.cam_name}', f'{tmp_uB.env_idx}')
                    
                        utils.print_color(f'step: {self.step}; buf rand len: {len(self.envBuf_rand)}')
                    

                
                ## -------------------------------------
                ## ---- 6. sample from the buffer ------
                ## (B,3,H,W); (B,4)
                ## check potental problem when in distrubted training
                
                ## img: tensor (B,3,128,128); act:tensor,
                imgs_start, imgs_goal, acts_gt, tasks_str_input, srb_info = self.sample_from_bufs()
                ## these imgs should be already crop and normalization, but still cpu tensor, (B, 3, H, W)
                assert imgs_start.shape[2:4] == self.input_img_size
                ## imgs_start: torch.Size([64, 3, 128, 128]), tasks_str_input: list of str
                ## srb_info['env_idxs']: np array
                # pdb.set_trace()

                if (self.debug and self.step < 500) or self.step % 100 == 0:
                    utils.print_color(f'{self.step}, iter_type: {self.iter_type}; batch len from buf: {len(tasks_str_input)}')
                    utils.print_color(f'envBuf_rand: {len(self.envBuf_rand)}')
                    utils.print_color(f'envBuf_vid: {len(self.envBuf_vid)}')

                
                ## --------------------------
                # pdb.set_trace()
                if (self.debug and self.step % 500 == 0) or self.step % 1000 == 0: # 30
                    n_vis = 3
                    utils.print_color(f'Save imgs sampled from mixed buffers.', c='b')
                    for i_sam in range(n_vis):
                        ## save st gl pari
                        tmp4 = (imgs_start[i_sam].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
                        tmp5 = (imgs_goal[i_sam].cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
                        fig1 = utils.plt_imgs_grid([tmp4, tmp5], caption='', texts=['start', 'goal'])
                        img_tmp_sg = plot2img(fig1,)
                        save_img_tr(img_tmp_sg, self.results_folder, 'imgs_stgl_from_buffer', 
                                tasks_str_input[i_sam], srb_info['cams_str'][i_sam], srb_info['env_idxs'][i_sam])
                        
                ## move to gpu
                imgs_start, imgs_goal, acts_gt = utils.to_device_tp(imgs_start, imgs_goal, acts_gt, device=device)

                # pdb.set_trace()
                assert imgs_start[0].shape == (3, 128, 128)
                ## ----------------------------

                ## ------- 7. compute loss --------
                with self.accelerator.autocast():
                    ## batch['obs']['img_obs_1'].shape: [32, 1, 3, 128, 128]
                    ## (B, self.model_act_horizon, 4)
                    batch = self.to_batch_dict(imgs_start, imgs_goal, acts_gt)

                    loss = self.gcp_model.compute_loss(batch)
                    infos = {}
                    
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.detach().item()

                    self.accelerator.backward(loss)
                    

            ## clip grad for diffusion policy
            accelerator.clip_grad_norm_(self.gcp_model.parameters(), 1.0)

            accelerator.wait_for_everyone()

            self.opt.step()
            self.opt.zero_grad()
            
            if self.lr_warmupDecay: # luo
                self.scheduler.step()

            accelerator.wait_for_everyone()

            self.step += 1
            
            ### Something only done in main Process (actually we only have one process)
            if accelerator.is_main_process:
                self.ema.update()

                if self.step % self.save_freq == 0 or self.step == 1:
                    label = self.step // self.label_freq * self.label_freq
                    self.save(label)

                if self.step % self.log_freq == 0 or self.step == 1:
                    if self.accelerator.scaler:
                        scale = self.accelerator.scaler.get_scale()
                    else:
                        scale = 'no'
                    infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                    print(f'{self.step}: {loss:8.4f} | {infos_str} | scale: {scale} | t: {timer():8.4f}')

                    metrics = {k:v.detach().item() for k, v in infos.items()}
                    metrics['train/it'] = self.step
                    metrics['train/loss'] = loss.detach().item()
                    metrics['train/lr'] = get_lr(self.opt)
                    metrics['train/loss_scale'] = scale
                    metrics['train/num_steps_in_env'] = self.num_steps_in_env
                    metrics['train/cnt_explore_suc'] = self.cnt_explore_suc
                    
                    metrics['buf/len_envBuf_rand'] = len(self.envBuf_rand)
                    metrics['buf/len_envBuf_vid'] = len(self.envBuf_vid)
                    ## NEW
                    metrics['explo/cnt_vid_rollouts'] = self.cnt_vid_rollouts
                    metrics['train/cnt_explore_suc_vsR'] = self.cnt_explore_suc
                    # pbar.set_description(f'loss: {total_loss:.4E}, loss scale: {scale:.1E}')
                    metrics.update( self.make_wandb_dict_per_tk() )

                    wandb.log(metrics, step=self.step)
                    print_gpu_utilization()

                if self.sample_freq and self.step % self.sample_freq == 0:
                    self.ema.ema_model.eval()
                    pass ## can do some evaluation here

                    


        accelerator.print('training complete')


    ## --------------------------------



    

    def env_get_preproc_imgs(self, tasks_str, cams_str, env_idxs,):
        '''
        return:
        a tensor of cropped imgs: (B, 3, 128, 128)
        '''
        # assert False, 'to be finised'
        assert len(tasks_str) == 1
        ## 2. get a batch of current state and images
        imgs_start = []
        for i_sam, tk in enumerate(tasks_str): # e.g. [101, 202]

            ## a list of np (h,w,3)
            img = self.env_list.render_an_env(tk, cams_str[i_sam], env_idxs[i_sam],)
            assert img.shape[:2] == self.input_img_size

            imgs_start.append(img)
            
            ## luotest
            if self.debug:
                save_img_tr(img, self.results_folder, 'render_imgs_1', 
                            tk, cams_str[i_sam], env_idxs[i_sam].item())

        if True: # preproc
            imgs_start = np.array(imgs_start) # B H W C
            # uint8 -> float, range[0,255] -> 0-1, np -> tensor, 
            imgs_start = self.rendered_imgs_preproc_fn(imgs_start,)
        
        # pdb.set_trace() # 1,3,128,128
        return imgs_start
    
    def env_get_preproc_img(self, tk, cam_name, env_idx):
        '''
        render an image of one env and convert to tensor that can input to the model
        tmp_img: (3,h,w)
        '''
        tmp_fn = partial(self.rendered_imgs_preproc_fn,) # crop_size=self.input_img_size)
        tmp_img = self.env_list.render_an_env_with_preproc(
            tk, cam_name, env_idx, imgs_preproc_fn=tmp_fn)
        ## 3,128,128
        assert tmp_img.shape[1:3] == self.input_img_size

        return tmp_img
    


    def h5_add_rand_act_episodes_to_Buf(self, start_ep_idx, end_ep_idx):
        '''
        ## ---------- NEW May 13 --------------
        Note that the args are relative to per task
        '''
        ## Load Data directly from the H5 and add to our rand buffer
        max_act_orn = -1
        min_act_orn = 1
        len_before = len(self.envBuf_rand)
        with h5py.File(self.randsam_file_path, 'r') as rs_file:
            for i_t, tk in enumerate(self.task_list):
                # for i_ep in range(self.num_init_rand_episodes):
                for i_ep in range(start_ep_idx, end_ep_idx):
                    ## out of range
                    if i_ep >= self.h5_total_num_ep_per_task:
                        assert f'{tk}/{i_ep}' not in rs_file
                        utils.print_color(f'\n\n\n{i_ep}: out-of-range\n\n\n', c='y')
                        break

                    fgroup = rs_file[f'{tk}/{i_ep}']
                    
                    ## will auto convert to one numpy, T,128,128,3
                    imgs_ep = fgroup['agentview_image'][:]
                    acts_ep = fgroup['action'][:] # -1 shorter than imgs_ep
                    
                    if self.debug:
                        max_act_orn = max( max_act_orn, acts_ep[:, 3:6].max().item() ) ## tk500: 0.111
                        min_act_orn = min( min_act_orn, acts_ep[:, 3:6].min().item() ) ## tk500: -0.111
                        if i_ep % 50 == 0 and i_ep != 0:
                            print(i_ep)
                    ## check orn range match
                    assert (acts_ep > self.act_min_np[None,] - 0.012 ).all() # (T,7) v.s. 1,7
                    assert (acts_ep < self.act_max_np[None,] + 0.012).all()
                    ## clip, since in initial version of randsam, I forget clipping
                    acts_ep = np.clip(acts_ep, a_min=self.act_min_np[None,], a_max=self.act_max_np[None,])

                    ## -------

                    ## convert to a list of tensor to fit in envBuf
                    ## batch of numpy to tensor
                    imgs_ep = self.rendered_imgs_preproc_fn(imgs_ep) # B,3,128,128
                    acts_ep = acts_ep.astype(np.float32)
                    acts_ep = torch.from_numpy(acts_ep)
                    # pdb.set_trace()
                    imgs_ep = list(torch.unbind(imgs_ep, dim=0)) # list of 3,128,128
                    acts_ep = list(torch.unbind(acts_ep, dim=0))
                    
                    assert len(imgs_ep) - 1 == len(acts_ep)
                    ## 
                    # pdb.set_trace() ## len act?
                    if not self.is_all_randsam_visited:
                        self.num_steps_in_env += len(acts_ep)


                    tmp_e_idx = self.env_list.seed_sets[tk][0]
                    ## should add tensor data
                    self.envBuf_rand.add_one_episode(
                        tk, self.env_list.camera_list[0], tmp_e_idx, imgs_ep, acts_ep,
                    )


        utils.print_color(f'[Rand Buf Size Before Load] ep {len_before}', c='y')
        utils.print_color(f'[Rand Buf Size After Load] ep {len(self.envBuf_rand)}', c='y')




    

    def sample_from_bufs(self):
        '''
        comment of this func can be removed
        sample data from the two buffers from training'''
        
        if len(self.envBuf_vid) == 0: # Initial, video execuation hasn't started
            n_rands, n_vids = self.buf_sample_batch_size, 0
            
            
            imgs_start, imgs_goal, acts_gt, tasks_str_input, srb_info = \
                self.envBuf_rand.sample_random_batch_seq(n_rands)

            ## imgs_start: B,3,H,W
            # pdb.set_trace() ## those images should be already tensor 3HW, please check
        elif len(self.envBuf_rand) == 0:
            n_rands, n_vids = 0, self.buf_sample_batch_size
            ## For ablation studies
            imgs_start, imgs_goal, acts_gt, tasks_str_input, srb_info = \
                self.envBuf_vid.sample_random_batch_seq(n_vids)
            assert self.init_rand_steps == -1

        else:

            if self.buf_sample_method == 'iter_bias_fix':
                if self.iter_type == 'rand-bias':
                    n_rands, n_vids = self.nums_buf_sample_rand 
                elif self.iter_type == 'vid-bias':
                    n_rands, n_vids = self.nums_buf_sample_vid_bias
                else:
                    raise NotImplementedError
            elif self.buf_sample_method == 'iter_bias_rand':
                probs = np.random.uniform(low=0, high=1, size=(self.buf_sample_batch_size,))
                if self.iter_type == 'rand-bias':
                    n_rands = (probs < self.r_prob_rand).sum()
                elif self.iter_type == 'vid-bias':
                    n_rands = (probs < self.r_prob_vid).sum()
                else:
                    raise NotImplementedError
                n_vids = self.buf_sample_batch_size - n_rands
            elif self.buf_sample_method == 'rand_prob':
                ## randomly sample from each buffers according to a prob, so the ratio is not fixed
                probs = np.random.uniform(low=0, high=1, size=(self.buf_sample_batch_size,))
                n_rands = (probs < self.buf_sample_randBuf_prob).sum()
                n_vids = self.buf_sample_batch_size - n_rands
                
            else:
                raise NotImplementedError
            

            r_b = self.envBuf_rand.sample_random_batch_seq(n_rands)
            v_b = self.envBuf_vid.sample_random_batch_seq(n_vids)

            tmp_fn = identity_tensor
            ## checked merge
            imgs_start, imgs_goal, acts_gt, tasks_str_input, srb_info = \
                merge_batch(r_b, v_b, imgs_preproc_fn=tmp_fn)
            
            # pdb.set_trace()
            assert imgs_goal.shape[2:4] == self.input_img_size
            assert len(imgs_goal) == self.buf_sample_batch_size
            assert len(srb_info['cams_str']) == self.buf_sample_batch_size

        # utils.print_color(f'rand buf_idx: {buf_idx}')

        return imgs_start, imgs_goal, acts_gt, tasks_str_input, srb_info






    
    def video_guided_explore(self):
        ''' run for all envs (whole dataset)
        1. generate videos for guidance
        2. execute following the video goal sequentially
        3. update vid buf
        '''
        ## reset env, actually I think no need to do it?
        self.reset_all_envs()
        buf_len_0 = len(self.envBuf_vid)
        utils.print_color(f'[Vid Exp] self.step {self.step}', c='y')
        
        ## rollout all 8 envs, ** one by one **, i.e., bs=1
        for _, batch in enumerate(self.dl_vid):
            # 1. get next data to gen videos and execute actions
            tasks_str, cams_str, env_idxs = batch
            env_idxs = env_idxs.cpu().numpy()
            assert len(tasks_str) == 1
            ## 2. init env
            self.env_list.init_1_given_env(tk_name=tasks_str[0], env_idx=env_idxs[0], is_rand=True)
            
            
            ## imgs_start: (B,3,128,128)
            imgs_start = self.env_get_preproc_imgs(tasks_str, cams_str, env_idxs)

            ## 2. Get the video for sequence guidance
            ## ---- a. fetch from buffer or b. directly run video prediction  -------
            if True:
                ## generate new videos every iterations, not using video buffer
                # assert not self.use_vid_replay_buf
                with torch.no_grad():
                    with self.accelerator.autocast():
                        ## already B,(T,3),H,W --> B,T,3,H,W
                        preds_video = self.video_model.forward(imgs_start.to(self.device), tasks_str).cpu()

                # pdb.set_trace()
                # save the predicted video to check
                if self.debug:
                    # save the pred videos to gif (3,h,w)
                    for i_sam in range(len(tasks_str)):
                        tmp2 = torch.cat( [ imgs_start[i_sam, None], preds_video[i_sam] ], dim=0 )
                        tmp2 = (tmp2.permute(0,2,3,1).numpy() * 255).astype(np.uint8)
                        save_gif_tr(tmp2, self.results_folder, 'render_imgs_1', 
                                        tasks_str[i_sam], cams_str[i_sam], env_idxs[i_sam].item())
                
                

                
            
            
            do_vis = True
            # e.g., B * [281 states]; B * [280 acts]
            batch_imgs_out_dense, batch_acts_out = self.envs_video_guided_execute(
                tasks_str, cams_str, env_idxs, imgs_start, preds_video, vis_rollout=do_vis)
            

            

            is_except = self.get_is_envs_exception(tasks_str, env_idxs)

            ## close env
            self.env_list.close_1_given_env(tk_name=tasks_str[0], env_idx=env_idxs[0])

            ## ------- update env buffer --------
            for i_sam, tk in enumerate(tasks_str): # e.g. [101, 202]
                if is_except[i_sam]:
                    # self.env_list.reset_env_instance(tk, env_idxs[i_sam])
                    continue

                self.envBuf_vid.add_one_episode(
                    tk, cams_str[i_sam], env_idxs[i_sam], 
                    batch_imgs_out_dense[i_sam], batch_acts_out[i_sam]
                )
        
        
        ## ---------------
        ## have finished sampling, executing, updating buf_vid for all envs (e.g., 12),
        buf_len_1 = len(self.envBuf_vid)
        utils.print_color(f'Finish Vid Explore, vid buf before: {buf_len_0}, after: {buf_len_1}')
        self.reset_all_envs() # this reset is a must


                

    def update_iter_type(self):
        ''' Important logic control func
        '''
        ## first larger number of rand explore
        if self.step < self.init_rand_steps:
            self.iter_type = 'rand-bias'

        elif self.step == self.init_rand_steps:
            self.rand_iter_cnt = 0
            
        ## then first into rand, switch to next if cnt reaches
        elif self.rand_iter_cnt == self.rand_cycle_steps:
            self.rand_iter_cnt = 0
            self.iter_type = 'vid-bias'
            utils.print_color(f'switch to {self.iter_type}', c='y')

        elif self.vid_iter_cnt == self.vid_cycle_steps:
            self.vid_iter_cnt = 0
            self.iter_type = 'rand-bias'
            utils.print_color(f'switch to {self.iter_type}', c='y')

        
        ## always random/pred explore
        if self.vid_cycle_steps == 0:
            self.iter_type = 'rand-bias'
        elif self.rand_cycle_steps == 0:
            self.iter_type = 'vid-bias'
        
        assert self.iter_type in ['rand-bias', 'vid-bias']
    
    def get_new_env_buffer(self, max_num_unitBufs):
        env_buf_config = dict(sample_act_seq_len=self.model_act_horizon)

        return Global_EnvReplayBuffer_Img(self.task_list, 
            max_num_unitBufs, self.max_len_uB, self.min_len_uB,
            self.env_list, self.render_img_size, env_buf_config)
        


    def get_is_envs_exception(self, tasks_str, env_idxs):
        '''get if env has exception, (act will have no effect)
        so that we do not add it to buffer,
        if exception, just waiting for reset.
        '''
        is_except = []
        for i_sam, tk in enumerate(tasks_str):
            env = self.env_list.get_an_env_ref(tk, env_idxs[i_sam])
            ## env._did_see_sim_exception ## assume no exception
            is_except.append(False)
        return is_except
    


    def envs_video_guided_execute(self, tasks_str, cams_str, env_idxs, 
                                 imgs_start, preds_video, vis_rollout=False):
        '''
        Collect rollout of a video for guided exploration
        We assume that all the envs are just reset, so no need to set an env
        Params:
            tasks_str: list of task name
            cams_str: list of cam name
            env_idxs: list of env ids
            imgs_start: (B, 3, H, W), should be already cropped
            preds_video: (B, 7, 3, H, W)
        Returns:
            imgs_out: list of tensors, collected imgs
            acts_out: list of tensors, collected actions
        '''
        # torch.Size([4, 7, 3, 128, 128])
        preds_video = preds_video.detach().to(self.device)

        # ----- Sanity Check -----
        assert imgs_start.shape[2:4] == self.input_img_size
        v_hzn = len(preds_video[0])
        assert v_hzn == self.video_model.video_future_horizon

        # ----- total roll out steps
        # require a lost of actions to complete a task, ~280
        max_steps = len(preds_video[0]) * self.max_acts_betw_vframes
        
        batch_imgs_out_dense = []
        batch_acts_out = [] # list of tensors (280,4)
        batch_vis_idxs = [] # a list of list of int to store some idxs to visualize
        # ------

        ## loop through several videos
        for i_sam, tk in enumerate(tasks_str):
            pred_v = preds_video[i_sam] # tensor (T=7, 3, H, W)
            img_st = imgs_start[i_sam:i_sam+1] # cpu tensor (B=1, 3, H, W)

            is_suc = False
            # imgs_out = [ img_st ]
            depths_out_1, depths_out_2 = [], [] # 1 no mask; 2 with mask
            
            # states_out = [ self.env_list.get_an_env_state(tk, env_idxs[i_sam]), ]
            print(f'video: {i_sam}')
            if self.debug:
                ## just to check state, can be removed
                img_test = self.env_get_preproc_img(tk, cam_name=cams_str[i_sam], env_idx=env_idxs[i_sam])
                utils.print_color(f'diff img_st: {torch.abs(img_test - img_st[0]).sum()}')
                

            # tensor imgs of one video rollout, [each img is (1, 3, 128, 128), ...]
            imgs_out_dense = [img_st,]

            acts_out = [] # gt acts, each elememt is a tensor [H,4] 
            texts_out = ['start img',]
            vis_idxs = [0,]
            
            ## Is this is a task that involves grasping?
            is_a_grasp_task = True
            
            ## total policy model forward = v_hzn * self.n_preds_betw_vframes
            ## 0. Roll out one video
            i_step = 0
            do_grasp = False
            num_acc_acts = 0
            ## -- loop rollout one video, g_idx: frame idx of the subgoal --
            for g_idx in range(v_hzn):
                
                img_goal = pred_v[None, g_idx] # (1,3,H,W)

                ## e.g.,[4,6], two-ends included
                n_preds = random.randint(*self.n_preds_betw_vframes)
                ## st -> p0, p0 -> p1, ..., 
                for i_p in range(n_preds):
                    # (1,3,H,W)
                    img_st = img_st.to(self.device)

                    ### ------ 4. forward policy model and get actions -------
                    ### -------- No Grad Forward ---------
                    with torch.no_grad():
                        
                        batch = self.to_batch_dict(img_st, img_goal, None)
                        # pdb.set_trace()
                        act_dict = self.ema.ema_model.predict_action(batch['obs'], use_ddim=True)
                        ## act is already truncated
                        act = act_dict['action'].cpu() # action_pred: full len
                    
                    
                    # (1, H, 4) -> (H, 4)
                    act = act[0]
                    assert len(act) == self.n_acts_per_pred

                    
                    ## Note that action should be normalized or Clamped
                    act = act.clamp(min=self.act_min, max=self.act_max)
                    # pdb.set_trace() # check clamp
                    
                    ### ---------- Grasping according to Depth, Apr 23, Part 1 ----------
                    if is_a_grasp_task:
                        if g_idx >= 0 and not do_grasp:
                            act[:, -1] = - self.close_grp_force # -0.98
                        elif do_grasp:
                            act[:, -1] = self.close_grp_force # +0.98
                    ### ----------
                    
                    ### Execute the first N actions
                    for i_a in range(self.n_acts_per_pred):
                        _,_,e_done,info = self.env_list.step_an_env(tk, env_idxs[i_sam], act[i_a].numpy())
                        # states_out.append( self.env_list.get_an_env_state(tk, env_idxs[i_sam]) )

                        ## current img obs, img_cur: (1,3,h,w)
                        img_cur = self.env_get_preproc_img(tk, cams_str[i_sam], env_idxs[i_sam])[None,]
                        imgs_out_dense.append(img_cur)

                        if True:
                            self.num_steps_in_env += 1

                    is_suc = e_done or is_suc
                    ## check img_cur: 1,3,128,128

                    img_st = torch.clone(imgs_out_dense[-1]) # img_cur
                    assert img_st.ndim == 4 and img_st.shape[0] == 1

                    # print( 'img_st', img_st.shape )
                    vis_idxs.append(len(imgs_out_dense)-1)

                    acts_out.append(act) # tensor [H, 4]

                    num_acc_acts += len(act)

                    texts_out.append(f'{i_step}:{g_idx+1}:acc_acts:{num_acc_acts}')

                    ## -------------------------------------------------------------
                    ## ------- Grasping according to Depth, Apr 23, Part 2 ---------
                    if is_a_grasp_task:
                        # self.render_depth_size = (160, 120)
                        self.grp_cam_name = 'gripper'



                        ## no need of negative sign here
                        grp_depth = self.env_list.render_an_env_with_depth(
                            tk, self.grp_cam_name, env_idxs[i_sam],)[1]
                        assert grp_depth.shape[:2] == (128, 128)
                        assert (grp_depth >= 0).all(), 'sanity check'
                        

                        
                        h, w = grp_depth.shape[:2] # here the shape is 128,128,1; different from MW
                        
                        ## Libero hyperparams
                        h_st = round(h * 0.75); h_e =  round(h * 0.82) ## luo3
                        w_st = round(w*0.35); w_e = round(w*0.65)


                        d_m = np.mean( grp_depth[h_st:h_e, w_st:w_e] ) # mean depth of the area under gripper
                        ## vis
                        depths_out_1.append( grp_depth )
                        grp_depth_2 = np.copy(grp_depth)
                        grp_depth_2[h_st:h_e, w_st:w_e] = 0
                        depths_out_2.append( grp_depth_2 )


                        ee_pos = self.env_list.get_an_env_obs(tk, env_idxs[i_sam])['robot0_eef_pos']
                        assert ee_pos.shape == (3,)

                        # substract order different from metaworld
                        z_diff = np.abs(ee_pos[2] - d_m).item() # z_diff should be 0.32 if no object under gripper

                        self.grasp_z_diff_limit = self.trainer_dict['grasp_z_diff_limit'] # 0.38
                        self.grasp_abs_z_limit = self.trainer_dict['grasp_abs_z_limit'] # 0.54


                        ## simple heuristic for gripper and if end effector z is close to the ground
                        if z_diff > self.grasp_z_diff_limit and ee_pos[2] < self.grasp_abs_z_limit \
                            and not do_grasp:
                        
                            
                            utils.print_color('Do Grasping', c='y')
                            do_grasp = True
                            assert self.control_mode == 'delta'
                            
                            n_acts_down = random.randint(self.n_acts_down_range[0], self.n_acts_down_range[1])
                            if self.act_down_val is None:
                                tk_idx = self.env_list.task_to_task_idx[tk] # e.g., 65
                                actd_rg = self.act_down_val_range_per_tk[tk_idx]
                                tmp_act_down_val = np.random.uniform(low=actd_rg[0], high=actd_rg[1], size=1).item()
                            else:
                                tmp_act_down_val = self.act_down_val
                            assert tmp_act_down_val <= 0

                            ## Do go down
                            act_down = torch.tensor([[0,0,tmp_act_down_val,0,0,0,0]] * n_acts_down) # e.g., [8, 4]
                            # pdb.set_trace()
                            for i_a in range(len(act_down)): # to align the temporal size
                                _,_,_,info = self.env_list.step_an_env(tk, env_idxs[i_sam], act_down[i_a].numpy() ) # move down
                                # states_out.append( self.env_list.get_an_env_state(tk, env_idxs[i_sam]) )
                                img_cur = self.env_get_preproc_img(tk, cams_str[i_sam], env_idxs[i_sam])[None,]
                                imgs_out_dense.append(img_cur)
                            
                            acts_out.append(act_down)
                            
                            ## Do grasping
                            act_grasp = torch.tensor( [[0,0,self.close_grp_act_down_val,0,0,0,self.close_grp_force]] * self.n_acts_close_grp )
                            for i_a in range(len(act_grasp)):
                                _,_,_,info = self.env_list.step_an_env(tk, env_idxs[i_sam], act_grasp[i_a].numpy()) # 0.98
                                # states_out.append( self.env_list.get_an_env_state(tk, env_idxs[i_sam]) )
                                img_cur = self.env_get_preproc_img(tk, cams_str[i_sam], env_idxs[i_sam])[None,]
                                imgs_out_dense.append(img_cur)

                            acts_out.append(act_grasp)
                            

                            ## renew img_st, also add these actions to buffer for training.
                            
                            # torch.Size([1, 3, 128, 128])
                            img_st = torch.clone(imgs_out_dense[-1])
                            
                            vis_idxs.append( len(imgs_out_dense)-1 )

                            num_acc_acts += (len(act_down) + len(act_grasp))

                            texts_out.append(f'*Grasp*{i_step}:{g_idx+1}:acc_acts:{num_acc_acts}')
                    else:
                        pass # not a grasp task, no need

                    

                    ## ------------------- Depth Ends ------------------------
                    ## ------------------------------------------------------- 



                    
                    i_step += 1
                
                
                ## ---- finished one video frame, aka, have run n_preds times -------
                if is_suc and self.is_stop_at_suc:
                    break
            

            ## -------------------------------------
            ## ---- finished one video rollout -----

            ## after act: tensor 2d, (total_steps, 4), e.g. (280+16+8, 4)
            acts_out = torch.cat(acts_out)
            # convert to a list of 1d tensor (4,)
            acts_out = list(torch.unbind(acts_out, dim=0))

            assert len(imgs_out_dense) == len(acts_out) + 1
            assert num_acc_acts == len(acts_out)
            assert max(vis_idxs) < len(imgs_out_dense)
            
            
            ## imgs_out_dense: a list of [1,c,h,w], 
            ## now convert to a list of [c,h,w] to be compatible with buffer
            imgs_out_dense = [img[0] for img in imgs_out_dense]

            ## acts_out: a tensor (T, 4)
            batch_imgs_out_dense.append(imgs_out_dense)
            batch_acts_out.append(acts_out)

            

            if is_suc:
                self.cnt_explore_suc += 1
                self.cnt_explo_suc_per_tk[tk] += 1
            self.cnt_vid_rollouts += 1
            self.cnt_vid_rout_per_tk[tk] += 1
            
            ## visualize
            if vis_rollout:
                ## remember now imgs_out_dense: a list of [3,h,w]
                ## cat a list of [1,3,H,W]
                imgs_out_vis = torch.cat([imgs_out_dense[i_v][None,] for i_v in vis_idxs], dim=0)

                full_v = torch.cat([imgs_out_dense[0][None,], pred_v.cpu()], dim=0) # 8,3,H,W
                
                fig1 = utils.plt_imgs_grid(full_v, caption='pred_video', texts=list(range(v_hzn+1)))
                fig2 = utils.plt_imgs_grid(imgs_out_vis, caption='env rollout', texts=texts_out)

                img12 = utils.cat_2_figs(fig1, fig2)
                save_img_tr(img12, self.results_folder, 'pred_n_rollout', tk, cams_str[i_sam], 
                            f'pr-{env_idxs[i_sam]}-out')
                
                if is_a_grasp_task:
                    fig3 = utils.plt_imgs_grid(depths_out_1, caption='gripper depth 1', texts=texts_out[1:])
                    img_r3 = plot2img(fig3,)
                    save_img_tr(img_r3, self.results_folder, 'depth', tk, cams_str[i_sam], 
                                f'{env_idxs[i_sam]}-1')

        # pdb.set_trace()

        ## --- finish the whole batch rollout ----
        
        
        return batch_imgs_out_dense, batch_acts_out
            



    def to_batch_dict(self, imgs_start, imgs_goal, acts_gt, goal_embed=None):
        '''
        transform to a dict to conform the diffusion policy's input
        '''
        assert imgs_start.ndim == 4
        
        batch = dict(
            obs={'img_obs_1': imgs_start[:, None, ...], 
                 'img_goal_1': imgs_goal[:, None, ...],},
        )
        if acts_gt is not None:
            assert acts_gt.ndim == 3 and acts_gt.shape[-1] == 7
            batch['action'] = acts_gt
        
        return batch


    
    def make_wandb_dict_per_tk(self,):
        '''create a result dict for logging'''
        tmp_metric = {}
        for tk in self.cnt_vid_rout_per_tk:
            tmp11 = f'explo/{tk}-cnt_vid_rollouts'
            tmp_metric[tmp11] = self.cnt_vid_rout_per_tk[tk]

            tmp12 = f"explo/{tk}-cnt_explore_suc_vsR"
            tmp_metric[tmp12] = self.cnt_explo_suc_per_tk[tk]

        return tmp_metric

    def init_wandb_metrics(self):
        ## init a metric for every task
        for tk in self.cnt_vid_rout_per_tk:
            tmp11 = f'explo/{tk}-cnt_vid_rollouts'
            wandb.define_metric(tmp11)
            tmp12 = f"explo/{tk}-cnt_explore_suc_vsR"
            wandb.define_metric(tmp12, step_metric=tmp11)


    




            
            




            

    