from typing import Dict
import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffuser.diffusion_policy.normalizer import ConstNormalizerGroup,LimitsConstNormalizer
from diffuser.diffusion_policy.base_image_policy import BaseImagePolicy
from diffuser.diffusion_policy.model.conditional_unet1d import ConditionalUnet1D
from diffuser.diffusion_policy.model.multi_image_obs_encoder import MultiImageObsEncoder
from diffuser.diffusion_policy.common.pytorch_util import dict_apply


class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            noise_scheduler_ddim: DDIMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            num_inference_steps_ddim=8,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            _target_=None, # placeholder
            cond_unet1d_config={}, ## May 16
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            cond_unet1d_config=cond_unet1d_config, # May16
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_ddim = noise_scheduler_ddim
        self.ddpm_var_temp = 1.0
        self.cond_unet1d_config = cond_unet1d_config
        
        ## do normalization using given constants
        self.normalizer = ConstNormalizerGroup(LimitsConstNormalizer,shape_meta,n_obs_steps)
        
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.num_inference_steps_ddim = num_inference_steps_ddim
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None, use_ddim=False,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        assert (condition_mask == False).all(), 'no given condition'
        model = self.model

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        
        
        scheduler = self.noise_scheduler_ddim if use_ddim else self.noise_scheduler
        n_inf_steps = self.num_inference_steps_ddim if use_ddim else self.num_inference_steps 

        # if not use_ddim:
            # kwargs['var_temp'] = self.ddpm_var_temp

        # set step values
        scheduler.set_timesteps(n_inf_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning, should be empty, nothing being assigned
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                       use_ddim=False) -> Dict[str, torch.Tensor]:
        """
        obs_dict: should be the obs subdict of batch; [No] must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        ## dict_keys(['img_obs_1', 'img_goal_1'])
        ## torch.Size([1, 1, 3, 128, 128])
        nobs = self.normalizer.normalize_d(obs_dict)
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            assert False
            

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            use_ddim=use_ddim,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        assert naction_pred.shape[2] == Da, 'sanity check'

        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        action_pred = action_pred.detach() # added by luo
        # pdb.set_trace() ## to check normalize 0.1; naction_pred[0, :4]; action_pred[0,:4]

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action, # actual len of timestep to execute
            'action_pred': action_pred # full predition
        }
        return result


    def compute_loss(self, batch: dict):
        '''
        batch:
            obs: dict
                - start (B,T,3,H,W), - goal, - agent pose (B,T,D)
            action: tensor
        '''
        # normalize input
        assert 'valid_mask' not in batch
        ## nobs['img_obs_1'].shape
        ## what should be in the batch and obs_dict?, norm to [-1, 1]
        nobs = self.normalizer.normalize_d(batch['obs']) # start img, goal img, agent pos

        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        assert batch['action'].shape[-1] == self.action_dim        

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            assert self.n_obs_steps == 1, 'temporally'
            # reshape B, T, ... to B*T
            ## for faster processing
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            # assert this_nobs['img_goal_1'].shape[0] == batch_size, "goal's hzn must be 1"
            
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            # img_obs(64) * n_obs + img_goal(64) = 128
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            assert False
            


        # Sample noise that we'll add to the action trajectory
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        

        loss = F.mse_loss(pred, target, reduction='none')
        
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss


    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.normalizer.to_device(*args, **kwargs)
        return self