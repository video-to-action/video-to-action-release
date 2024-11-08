import sys, pdb
if __name__ == '__main__':
    sys.path.insert(len(sys.path)-3, '.')

from flowdiffusion.flowdiffusion.goal_diffusion import GoalGaussianDiffusion
from flowdiffusion.flowdiffusion.unet import Unet_Libero
from transformers import CLIPTextModel, CLIPTokenizer
from einops import rearrange
import torch, pdb
import numpy as np
from diffuser.models.video_model import Video_PredModel

def lb_get_video_model_gcp_v2(ckpts_dir='../ckpts/metaworld', milestone=24, flow=False, 
                        timestep=100, g_w=2.0, sample_per_seq=8, target_size=(128,128), 
                        model_version='luo_128_v0',
                        **kwargs):
    '''
    For Libero
    '''
    if  model_version == 'luo_128_v0':
        unet = Unet_Libero()
    else:
        raise NotImplementedError

    pretrained_model = "openai/clip-vit-base-patch32"
    ## isinstance(tokenizer, nn.Module) -> False
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    ## isinstance(text_encoder, nn.Module) -> True
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    # sample_per_seq = 8
    # target_size = (128, 128)
    channels = 3 if not flow else 2

    ## timestep:20, 
    ## check timestep and load dir

    ## the so-called channels is different in the two model
    diffusion = GoalGaussianDiffusion(
        channels=channels*(sample_per_seq-1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=timestep,
        loss_type='l2',
        objective='pred_v',
        beta_schedule='cosine',
        min_snr_loss_weight=True,
        guidance_weight=g_w,
    )

    video_model = Video_PredModel(
        diffusion,
        tokenizer,
        text_encoder,
        single_img_channels=channels,
        results_folder=ckpts_dir,
    )
    
    video_model.load_trained_model(milestone)
    video_model.requires_grad_(False)
    video_model.eval()

    return video_model


    
if __name__ == '__main__':

    tmp = dict(
            ckpts_dir='./ckpts/',
            milestone=24,
            timestep=20,
            g_w=0,
            cls_free_prob=0.0,
        )
    video_model = lb_get_video_model_gcp_v2(**tmp)
    pdb.set_trace()