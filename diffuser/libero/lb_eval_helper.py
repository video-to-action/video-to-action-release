from typing import List, Dict
from environment.libero.lb_env_v3 import LiberoEnvList_V3
import torch, pdb
import numpy as np
import os.path as osp
import diffuser.utils as utils
from ema_pytorch import EMA
from diffuser.models.video_model import Video_PredModel
from diffuser.libero.lb_online_trainer_v7 import LB_Online_Trainer_V7
from environment.libero.lb_utils import lb_merged_str_idx_tk_1

LB_1_VIDEO_PRED = []

class LB_DP_Eval(object):
    def __init__(self,
                 gcp_model,
                 ema: EMA, # EMA of diffusion policy
                 video_model: Video_PredModel,
                 trainer: LB_Online_Trainer_V7,

                 env_list: LiberoEnvList_V3,
                 task_list: List[str],
                 cam_list: List[str],

                 valid_seeds,
                 max_episode_steps, # 400
                 render_img_size,
                 rendered_imgs_preproc_fn,
                 is_video_ddim: bool,
                 is_dp_ddim: bool,
                 eval_n_preds_betw_vframes: int,

                 save_path,
                 vid_use_autocast=True,
                 num_vid_pred_per_ep=5,
                 use_vid_first_n_frames=2,
                 device='cuda',
                 ):
        self.gcp_model = gcp_model
        self.ema = ema
        self.video_model = video_model
        self.trainer = trainer

        self.env_list = env_list
        self.task_list = task_list
        self.cam_list = cam_list

        self.valid_seeds = valid_seeds
        self.max_episode_steps = max_episode_steps
        self.render_img_size = render_img_size
        self.rendered_imgs_preproc_fn = rendered_imgs_preproc_fn

        ## inherit from the trainer
        self.n_acts_per_pred = self.trainer.n_acts_per_pred
        self.input_img_size = self.trainer.input_img_size
        self.accelerator = self.trainer.accelerator

        ## -----
        self.video_model.ema.ema_model.is_ddim_sampling = is_video_ddim
        self.is_dp_ddim = is_dp_ddim
        self.eval_n_preds_betw_vframes = eval_n_preds_betw_vframes

        self.num_vid_pred_per_ep = num_vid_pred_per_ep
        self.use_vid_first_n_frames = use_vid_first_n_frames

        ## logs/{envlist_name}/diffusion/{config_name}
        self.save_path = save_path # save results
        self.vid_use_autocast = vid_use_autocast
        self.device = device
        # assert len(self.task_list) == len(self.cam_list) == 1
        assert max_episode_steps == 500
        # pdb.set_trace() # check env_list seed
        from diffuser.datasets.img_utils import imgs_preproc_simple_noCrop_v1
        assert self.rendered_imgs_preproc_fn == imgs_preproc_simple_noCrop_v1

        ## set seed for video gen, to make video teaser
        self.pre_vid_gen_fn = lambda **kargs: None
        self.pvG_fn_args = {}
        self.after_vid_gen_fn = lambda **kargs: None
        self.avG_fn_args = {}
        self.is_stop_at_suc = True

    
    def run_evals(self, vis_gif=True, cur_num_iters=''):
        '''
        Run all the evaluations.
        cur_num_iters: used for mark dirname when eval during training
        Returns:
        a dict
        '''
        is_sucs_all = []
        is_sucs_per_tk: Dict[str, list] = {}
        run_times_all = []
        run_times_per_tk = {}
        for i_sam, tk in enumerate(self.task_list):
            is_sucs_per_tk[tk] = []
            run_times_per_tk[tk] = []
            for cam_name in self.cam_list:
                if cam_name in ['agentview_image', 'agentview_rgb']:
                    cam_name = 'agent'
                for env_seed in self.valid_seeds:
                    
                    tmp_env_idx = self.env_list.seed_sets[tk][0]
                    env = self.env_list.init_1_given_env(tk, 
                         env_idx=tmp_env_idx, e_seed=env_seed)
                    
                    ## pred_v is the last video
                    is_suc, imgs_out, run_time, all_full_pred_v, img12 = self.eval_1_env(env, tk, cam_name)
                    self.env_list.close_1_given_env(tk, tmp_env_idx)
                    

                    is_sucs_all.append(is_suc)
                    is_sucs_per_tk[tk].append(is_suc)

                    run_times_all.append(run_time)
                    run_times_per_tk[tk].append(run_time)
                    
                    ## save gif
                    if vis_gif:
                        mg_str = lb_merged_str_idx_tk_1(self.env_list, tk)

                        sub_dir = f'{mg_str}-{cam_name}'
                        if cur_num_iters != '':
                            sub_dir += f'-{cur_num_iters}'
                        parent_path = osp.join(self.save_path, sub_dir)
                        ## save to gif
                        # gif_fname = f'{env_seed:03d}-{is_suc}.gif'
                        # utils.save_gif(imgs_out, parent_path, gif_fname, dr=0.05)
                        ## save to mp4
                        mp4_fname = f'{env_seed:03d}-{is_suc}.mp4'
                        utils.save_imgs_to_mp4(imgs_out, f'{parent_path}/{mp4_fname}', fps=50)


                        ## save pred video to mp4
                        for i_v in range(len(all_full_pred_v)):
                            ## tensor (8, 3, 128, 128), 0-1, permute and to np
                            v_imgs = list( all_full_pred_v[i_v].permute(0,2,3,1).numpy() )
                            mp4_fname_2 = f'{env_seed:03d}-{is_suc}-predv-{i_v}.mp4'
                            utils.save_imgs_to_mp4(v_imgs, f'{parent_path}/{mp4_fname_2}', fps=3)


                        ## save concat img
                        img12_path = osp.join(parent_path, f'{env_seed:03d}-{is_suc}.png')
                        utils.save_img(img12_path, img12)
                    

        ## finished all rollout
        suc_rate_per_tk = {}
        for i_sam, tk in enumerate(self.task_list):
            suc_rate_per_tk[tk] = np.mean(is_sucs_per_tk[tk]).item()



        return dict(suc_rate=np.mean(is_sucs_all).item(), 
                    num_evals=len(is_sucs_all),
                    n_seeds=len(self.valid_seeds),
                    suc_rate_per_tk=suc_rate_per_tk, # a dict of float
                    is_sucs_per_tk=is_sucs_per_tk, # per task
                    is_sucs_all=is_sucs_all,
                    run_times_all=run_times_all,
                    run_times_per_tk=run_times_per_tk,
                    seeds=self.valid_seeds,
                    )




    def eval_1_env(self, env, tk, cam_name):
        '''
        rollout only one env
        Returns:
            - is_suc:
            - imgs_out_dense: all obs imgs from env
            - run_time:
            - all_full_pred_v: all pred videos in a list
            - img12: put pred video and rollout obs imgs onto one large img
        '''

        self.debug = True
        self.ema.ema_model.eval()
        self.video_model.ema.ema_model.eval()
        
        timer = utils.Timer()
        is_suc = False

        ## -------------------------------------------------
        ## 1. prepare start img and feed to the video model
        ## -------------------------------------------------
        utils.print_color(f'cam_name: {cam_name}')
        
        ## shape: (240, 320, 3)
        img_r = self.env_list.render_a_given_env(env, cam_name,)
        assert type(img_r) == np.ndarray

        ## (1, 3, 128, 128)
        img_init_tensor = self.rendered_imgs_preproc_fn(img_r[None,],)
        assert img_init_tensor.shape[2:4] == self.input_img_size
        assert img_init_tensor.ndim == 4 and img_init_tensor.shape[1] == 3

        tasks_str = [tk,] # for the video model

        ## -------------------------------------------------
        ## 2. Prepare
        ## -------------------------------------------------

        # ----- Sanity Check -----
        assert img_init_tensor.shape[2:4] == self.input_img_size
        v_hzn = self.video_model.video_future_horizon # len(preds_video[0]) # 7

        # cpu tensor (B=1, 3, H, W)
        img_st = img_init_tensor

        is_suc = False
        
        ## save all the rollout imgs
        # tensor imgs of one video rollout, [each img is (1, 3, 128, 128), ...]
        imgs_out_dense = [img_st,]
        acts_out = [] # gt acts, each elememt is a tensor [H,4] 
        texts_out = ['start img',]
        vis_idxs = [0,]
        all_full_pred_v = [] # store all the videos

        ## total policy model forward = v_hzn * self.n_preds_betw_vframes
        i_step = 0
        
        num_acc_acts = 0
        
        
        ## -------------------------------------------------
        ## 3. for loop rollout
        ## -------------------------------------------------
        cnt_vid_pred = 0
        if tk in LB_1_VIDEO_PRED: ##
            num_vid_ppp = 1
        else:
            num_vid_ppp = self.num_vid_pred_per_ep

        num_total_frames = (num_vid_ppp - 1) * self.use_vid_first_n_frames + v_hzn
        ## -- loop rollout one video, g_idx: frame idx of the subgoal --
        for fr_idx in range(num_total_frames):
            
            if cnt_vid_pred < num_vid_ppp:
                if fr_idx == 0 or g_idx == self.use_vid_first_n_frames - 1:
                    
                    self.pre_vid_gen_fn(**self.pvG_fn_args)
                    with torch.no_grad():
                        if self.vid_use_autocast:
                            with self.accelerator.autocast():
                                ## already B,(T,3),H,W --> B,T,3,H,W
                                preds_video = self.video_model.forward(img_st.to(self.device), tasks_str)
                        else:
                            preds_video = self.video_model.forward(img_st.to(self.device), tasks_str)
                    
                    self.after_vid_gen_fn(**self.avG_fn_args)

                    assert len(preds_video) == 1
                    # torch.Size([1, 7, 3, 128, 128])
                    preds_video = preds_video.detach() # .to(self.device)
                    pred_v = preds_video[0] # tensor (T=7, 3, H, W)
                    
                    all_full_pred_v.append( torch.cat([img_st, pred_v.cpu()], dim=0) )
                    cnt_vid_pred += 1
                    g_idx = 0
                else:
                    g_idx += 1
            else:
                # use old pred video, but next frame
                g_idx += 1
            
            ## ---- set goal -----
            img_goal = pred_v[None, g_idx] # (1,3,H,W)
            ## sanity check
            if num_total_frames == fr_idx:
                assert g_idx == v_hzn - 1
            print(f'Libero fr_idx: {fr_idx}, g_dix: {g_idx}')




            ## Eval: 5; Train: e.g.,[4,6], two-ends included
            n_preds = self.eval_n_preds_betw_vframes
            assert type(n_preds)  == int

            ## st -> p0, p0 -> p1, ..., 
            for i_p in range(n_preds):
                # (1,3,H,W)
                img_st = img_st.to(self.device)

                ### ------ 4. forward policy model and get actions -------
                ### -------- No Grad Forward ---------
                with torch.no_grad():
                    
                    # obs_dict['obs']['img_obs_1'].shape # [1, 1, 3, 128, 128]
                    batch = self.trainer.to_batch_dict(img_st, img_goal, None)

                    act_dict = self.ema.ema_model.predict_action(batch['obs'], use_ddim=self.is_dp_ddim)
                    ## act is already truncated
                    act = act_dict['action'].cpu() # action_pred: full len
                
                # (1, H, 4) -> (H, 4)
                act = act[0]
                
                assert len(act) == self.n_acts_per_pred

                ## Note that action should be normalized or Clamped
                act = act.clamp(min=self.trainer.act_min, max=self.trainer.act_max)
                assert act.shape[-1] == 7

                ### Execute the first N actions
                for i_a in range(self.n_acts_per_pred):

                    _,_,e_done,info = env.step(act[i_a].numpy())

                    ## current img obs, img_cur: (1,3,h,w)
                    # img_cur = self.trainer.env_get_preproc_img(tk, cams_str[i_sam], env_idxs[i_sam])[None,]
                    img_cur = self.env_list.render_a_given_env(
                        env, cam_name=cam_name, # resolution=self.render_img_size
                    )
                    ## tensor 1,3,128,128
                    img_cur = self.rendered_imgs_preproc_fn(img_cur[None,],) # self.input_img_size)
                    imgs_out_dense.append(img_cur)

                    is_suc = bool(e_done) or is_suc
                
                

                img_st = torch.clone(imgs_out_dense[-1]) # img_cur
                assert img_st.ndim == 4 and img_st.shape[0] == 1

                vis_idxs.append(len(imgs_out_dense)-1)
                acts_out.append(act) # tensor [H, 4]
                num_acc_acts += len(act)
                texts_out.append(f'{i_step}:v{len(all_full_pred_v)}:g{g_idx+1}:acc_acts:{num_acc_acts}')

                
                i_step += 1
            
            ## ---- finished one video frame, i.e., have run n_preds times for one frame -------
            if is_suc and self.is_stop_at_suc:
                break
        
        ## -------------------------------------------------------------
        



        run_time = timer()

        ## now convert to a list of [c,h,w] to be compatible with buffer
        imgs_out_dense = [img[0] for img in imgs_out_dense]
        
        ## Code for Visualization
        if True:
            imgs_out_vis = torch.cat([imgs_out_dense[i_v][None,] for i_v in vis_idxs], dim=0)

            figs_all = []
            for full_v in all_full_pred_v:
                fig1 = utils.plt_imgs_grid(full_v, caption='pred_video', texts=list(range(v_hzn+1)), max_n_cols=8)
                figs_all.append(fig1)
            fig2 = utils.plt_imgs_grid(imgs_out_vis, caption='env rollout', texts=texts_out, max_n_cols=8)
            figs_all.append(fig2)

            img12 = utils.cat_n_figs(figs_all)


        ## a list of np (128,128,3), float32, 0-1
        imgs_out_dense = [img.permute(1,2,0).cpu().numpy() for img in imgs_out_dense]
        assert imgs_out_dense[0].ndim == 3
        print('imgs_out_dense:', len(imgs_out_dense), imgs_out_dense[0].shape)
        print('is_suc:', is_suc)
        
        
        return is_suc, imgs_out_dense, run_time, all_full_pred_v, img12



