import diffuser.utils as utils
from omegaconf import OmegaConf
from diffuser.datasets import image_minmax_01_f, mw_sawyer_action_minmax_f, lb_action_minmax_orn01_f, lb_action_minmax_f, thor_action_minmax_dim4_f, cal_action_minmax_f, cal_abs_action_minmax_f, tk_emb_minmax_f
from diffuser.diffusion_policy.common.python_utils import read_yaml

from diffuser.diffusion_policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from ema_pytorch import EMA
import pdb

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("image_minmax_01", image_minmax_01_f, replace=True)
OmegaConf.register_new_resolver("mw_action_minmax", mw_sawyer_action_minmax_f, replace=True)
OmegaConf.register_new_resolver("lb_action_minmax_orn01", lb_action_minmax_orn01_f , replace=True)
OmegaConf.register_new_resolver("lb_action_minmax", lb_action_minmax_f , replace=True)
OmegaConf.register_new_resolver("thor_action_minmax_dim4", thor_action_minmax_dim4_f, replace=True)

OmegaConf.register_new_resolver("cal_action_minmax", cal_action_minmax_f, replace=True)
OmegaConf.register_new_resolver("cal_abs_action_minmax", cal_abs_action_minmax_f, replace=True)

OmegaConf.register_new_resolver("tk_emb_minmax", tk_emb_minmax_f, replace=True)



class Init_Diffusion_Policy:
    '''a wrapper class to init diffusion policy'''

    def __init__(self, args) -> None:
        self.args = args
        fname = args.model_yl_path # a yaml path

        self.all_conf = read_yaml(fname)
        self.policy_conf = policy_conf = self.all_conf.policy
        # print('policy_conf', policy_conf)
        self.override_cfg()
        self.check_cfg()
        

        ## ------ Create the Policy -------
        ## 1. two schedulers
        ns_conf = policy_conf.noise_scheduler
        ns_cfg = utils.Config(
            _class=ns_conf._target_,
            verbose=False,
            **ns_conf,
        )
        noise_scheduler = ns_cfg()

        ns_ddim_conf = policy_conf.noise_scheduler_ddim
        ns_ddim_cfg = utils.Config(
            _class=ns_ddim_conf._target_,
            verbose=False,
            **ns_ddim_conf,
        )
        noise_scheduler_ddim = ns_ddim_cfg()

        ## 2.1 init rgb model for obs_encoder
        rm_conf = policy_conf.obs_encoder.rgb_model
        rm_cfg = utils.Config(
            _class=rm_conf._target_,
            verbose=False,
            **rm_conf,
        )
        rgb_model = rm_cfg()

        ## 2.2 init obs_encoder
        oe_conf = policy_conf.obs_encoder # MultiImageObsEncoder
        oe_cfg = utils.Config(
            _class=oe_conf._target_,
            verbose=False,
            **oe_conf,
        )
        del oe_cfg._dict['rgb_model']
        obs_encoder = oe_cfg(rgb_model=rgb_model)
        # print(policy_cfg.shape_meta)

        ## 3. init diffusion policy itself
        dp_conf = policy_conf
        dp_cfg = utils.Config(
            _class=dp_conf._target_,
            verbose=False,
            **dp_conf,
        )
        del dp_cfg._dict['noise_scheduler']
        del dp_cfg._dict['noise_scheduler_ddim']
        del dp_cfg._dict['obs_encoder']

        self.diffusion_policy: DiffusionUnetImagePolicy 
        self.diffusion_policy = dp_cfg(noise_scheduler=noise_scheduler, 
                    noise_scheduler_ddim=noise_scheduler_ddim, obs_encoder=obs_encoder)
        
        # self.create_ema()
    
    def override_cfg(self):
        self.policy_conf.horizon = self.args.trainer_dict['model_act_horizon']
        self.policy_conf.n_action_steps = self.args.trainer_dict['n_acts_per_pred']
        # self.policy_conf.n_obs_steps # 1 default

    
    def check_cfg(self):
        '''check if two config align'''
        assert self.args.input_img_size == tuple(self.all_conf.image_shape[1:])



