from .conditional_unet1d import ConditionalUnet1D as Unet1D_diffusion
from .perceiver import PerceiverResampler
from .vis_encoder import ResNet18Encoder
from .transformer_for_diffusion import TransformerForDiffusion
from torch import nn
from einops import rearrange, repeat
import torch

class Unet1D(nn.Module):
    def __init__(self, action_space=4, obs_steps=2):
        super(Unet1D, self).__init__()

        self.perceiver = PerceiverResampler(
            dim=512, depth=2
        )
        self.unet = Unet1D_diffusion(
            input_dim=action_space,
            local_cond_dim=None,
            global_cond_dim=512+512*obs_steps,
            diffusion_step_embed_dim=256,
            down_dims=[256,512,1024],
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False
        )
        self.resnet = ResNet18Encoder()

        self.last_obs = None

    def encode_obs_features(self, obs):
        # obs.shape = (b, f, c, h, w)
        f = obs.shape[1]
        obs = rearrange(obs, 'b f c h w -> (b f) c h w')
        obs = rearrange(self.resnet(obs), '(b f) c -> b (f c)', f=f)
        self.obs_features = obs

    def forward(self, x, t, task_embed):
        action, obs = x
        
        task_embed = self.perceiver(task_embed).mean(dim=1)
        # obs.shape = (b, f, c, h, w)
        if self.last_obs is None or not torch.allclose(self.last_obs, obs):
            self.encode_obs_features(obs)
            self.last_obs = obs

        global_cond = torch.cat([task_embed, self.obs_features], dim=1)
        return self.unet(action, t, global_cond=global_cond)

    
class TransformerNet(nn.Module):
    def __init__(self, action_space=4, obs_steps=2):
        super(TransformerNet, self).__init__()

        self.perceiver = PerceiverResampler(
            dim=512, depth=2
        )
        self.model = TransformerForDiffusion(
            input_dim=4,
            output_dim=4,
            horizon=10,
            n_obs_steps=3,
            cond_dim=512,
            n_cond_layers=2,
            n_layer=8,
            n_head=8,
            n_emb=384,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
        )
        self.resnet = ResNet18Encoder()

        self.last_obs = None

    def encode_obs_features(self, obs):
        # obs.shape = (b, f, c, h, w)
        f = obs.shape[1]
        obs = rearrange(obs, 'b f c h w -> (b f) c h w')
        obs = rearrange(self.resnet(obs), '(b f) c -> b f c', f=f)
        self.obs_features = obs

    def forward(self, x, t, task_embed):
        action, obs = x
        
        task_embed = self.perceiver(task_embed).mean(dim=1, keepdim=True)
        # obs.shape = (b, f, c, h, w)
        if self.last_obs is None or not torch.allclose(self.last_obs, obs):
            self.encode_obs_features(obs)
            self.last_obs = obs

        cond_feat = torch.cat([task_embed, self.obs_features], dim=1)

        return self.model(action, t, cond_feat)

if __name__ == "__main__":
    pass
    


