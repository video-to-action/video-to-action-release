import torch.nn as nn
import torch, pdb
from ema_pytorch import EMA
from diffuser.models.helpers import get_no_dash_tasks_str, get_no_underscore_tasks_str
from pathlib import Path
from flowdiffusion.flowdiffusion.goal_diffusion import GoalGaussianDiffusion
from einops import rearrange

class Video_PredModel(nn.Module):
    '''a wrap class only for video prediction inference'''
    def __init__(
        self,
        diffusion_model: GoalGaussianDiffusion, # not ema here
        tokenizer, 
        text_encoder,
        single_img_channels = 3,
        results_folder = './results',
    ):
        super().__init__()
        
        assert isinstance(diffusion_model, nn.Module)
        assert not isinstance(diffusion_model, EMA)
        ## create an ema model to fit the load operation, just a wrapper
        self.ema = EMA(diffusion_model, beta=0.995, update_every=10)
        
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.single_img_channels = single_img_channels # 3

        self.image_size = diffusion_model.image_size

        self.results_folder = Path(results_folder)
        # self.results_folder.mkdir(exist_ok = True)
        self.video_future_horizon = round(diffusion_model.channels/single_img_channels)
        # pdb.set_trace()
        

    def load_trained_model(self, milestone):
        '''load the ema model'''
        
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'),)        

        self.ema.load_state_dict(data["ema"], strict=True)
        
        if 'version' in data:
            print(f"loading from version {data['version']}")

    
    def encode_batch_text(self, batch_text):
        batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.device)
        ## torch.Size([1, 4=n_words+2, 512])
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state
        return batch_text_embed
    
    def sample(self, x_conds, tasks):
        assert x_conds.shape[0] == len(tasks)
        
        device = self.device
        bs = x_conds.shape[0]
        x_conds = x_conds.to(device)
        
        tasks = get_no_dash_tasks_str(tasks)
        
        tasks = get_no_underscore_tasks_str(tasks)
        

        ## get encoded text features 
        tasks = self.encode_batch_text(tasks).to(device)
        
        output = self.ema.ema_model.sample(batch_size=bs, x_cond=x_conds, task_embed=tasks)

        # reshape to a video, e.g., B,7,3,128,128
        output = rearrange( output, "b (f c) h w -> b f c h w", c=self.single_img_channels ).detach()

        return output
    
    def forward(self, x_conds, tasks):
        return self.sample(x_conds, tasks)
    
   
    

    @property
    def device(self):
        return next(self.ema.parameters()).device
        