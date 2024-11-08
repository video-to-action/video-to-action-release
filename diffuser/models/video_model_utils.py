import sys, pdb
if __name__ == '__main__':
    sys.path.insert(len(sys.path)-3, '.')
from flowdiffusion.flowdiffusion.goal_diffusion import GoalGaussianDiffusion, Trainer
from flowdiffusion.flowdiffusion.unet import UnetMW as Unet
from flowdiffusion.flowdiffusion.unet import UnetMWFlow as Unet_flow
from flowdiffusion.flowdiffusion.unet import UnetThor as Unet_thor
from flowdiffusion.flowdiffusion.unet import UnetBridge as Unet_bridge
from transformers import CLIPTextModel, CLIPTokenizer
from einops import rearrange
import torch, pdb
import numpy as np
from diffuser.models.video_model import Video_PredModel

def get_video_model_gcp(ckpts_dir='../ckpts/metaworld', milestone=24, flow=False, 
                        timestep=100, g_w=2.0, **kwargs):
    unet = Unet_flow() if flow else Unet()
    pretrained_model = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    sample_per_seq = 8
    target_size = (128, 128)
    channels = 3 if not flow else 2

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
    
    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=[1],
        valid_set=[1],
        results_folder = ckpts_dir,
        fp16 =True,
        amp=True,
    )
    
    trainer.load(milestone)

    return trainer


def get_video_model_gcp_v2(ckpts_dir='../ckpts/metaworld', milestone=24, flow=False, 
                        timestep=100, g_w=2.0, sample_per_seq=8, target_size=(128,128), **kwargs):
    '''currently using, Feb 29
    This is for MetaWorld Only
    '''
    
    unet = Unet_flow() if flow else Unet()
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
            ckpts_dir='./ckpts/metaworld',
            milestone=24,
            timestep=20,
            g_w=0,
            cls_free_prob=0.0,
        )
    video_model = get_video_model_gcp_v2(**tmp)
    pdb.set_trace()