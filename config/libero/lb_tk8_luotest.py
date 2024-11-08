import os.path as osp
from diffuser.utils import watch
from diffuser.datasets import LB_ACTION_MIN, LB_ACTION_MAX
from diffuser.libero.lb_constants import LB_GRASP_actdown_value_range_1

"""
NOTE:
This is just a template to test if the code works
"""

#------------------------ base ------------------------#

config_fn = osp.splitext(osp.basename(__file__))[0]
diffusion_args_to_watch = [
    ('prefix', ''),
    ('config_fn', config_fn),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]


plan_args_to_watch = [
    ('prefix', ''),
    ('config_fn', config_fn),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
]

act_hzn = 16

base = {
    'dataset': "libero-8tk-65to72-v3",
    'diffusion': {
        'config_fn': '',

        'model_yl_path': 'config/diff_policy/lb_train_diffusion_unet_image_orn10.yaml',

        'vid_diffusion': dict(
            ckpts_dir='./ckpts/libero/libero_ep20_bs12_aug',
            milestone=180000,
            timestep=100,
            g_w=0,
            cls_free_prob=0.0,
            sample_per_seq=8,
        ),
        
        'input_img_size': (128, 128),
        'render_img_size': (128, 128),

        ## dataset
        'loader': 'diffuser.libero.lb_online_dataset.LB_Online_Dataset',
        'dataset_config': dict(
            act_min_max=(LB_ACTION_MIN, LB_ACTION_MAX),
            combo_type='all',
        ),
        
        'allow_tf32': True,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'trainer_type': 'v7',
        'do_train_resume': False,

        'trainer_dict': dict(


            num_init_rand_Ep_per_tk=50,
            envBuf_max_num_uB_rand=1200,
            envBuf_max_num_uB_vid=600,
            max_len_uB=700,
            min_len_uB=30,

            is_stop_at_suc=False,

            
            model_act_horizon=act_hzn,

            init_rand_steps=100, ## TODO: 10000
            rand_cycle_steps=100,
            vid_cycle_steps=400,
            use_env_rand_reset=True, # x

            video_explo_freq=200,
            rand_explo_freq=500,
            rand_explo_num_Ep_per_tk=2,


            ## guided rollout
            n_acts_per_pred=8,
            n_preds_betw_vframes=(4,6),
            

            batch_size=4, # x
            batch_size_v=1,
            buf_sample_batch_size=64,
            buf_sample_ratio_rand=[0.75, 0.25], 
            buf_sample_ratio_vid=[0.25, 0.75], 

            buf_sample_method='rand_prob',
            buf_sample_randBuf_prob=0.3,


            enable_noExp=True,
            noExp_start_buf_len_rand=500,
            noExp_start_buf_len_vid=500,
            Exp_noExp_rand=(1000, 1000),
            Exp_noExp_vid=(1000, 1000),

            n_acts_down_range=(16, 16),
            n_acts_close_grp=8,


            act_down_val=None,
            act_down_val_range_per_tk=LB_GRASP_actdown_value_range_1,

            close_grp_force=0.98,
            close_grp_act_down_val=0,
            
            rand_explo_type='from_h5',
            randsam_filename='lb_randsam_8tk_perTk500.hdf5',
            grasp_z_diff_limit=0.36,
            grasp_abs_z_limit=0.56,
            

        ),
        'loss_type': 'l2',
        'n_train_steps': 2e5,

        'gradient_accumulate_every': 1,
        
        'opt_params': dict(
            lr=1.0e-4,
            betas=[0.95, 0.999],
            eps=1.0e-8,
            weight_decay=1.0e-6,
        ),
        
        'ema_params': dict(
            update_after_step=0,
            inv_gamma=1.0,
            power=0.75,
            min_value=0.0,
            # max_value=0.9999,
            update_every=1,
            include_online_model=False,
        ),

        'save_freq': 1000,
        'sample_freq': 5000,
        'log_freq': 100,
        'n_saves': 5, # num of checkpoint to save

        'num_render_samples': 2, 
        'save_parallel': False,
        'n_reference': 2,
        'n_samples': 1,

    },

    'plan': {
        'config_fn': '',
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',
        'diffusion_epoch': 'latest', #

    },

}
