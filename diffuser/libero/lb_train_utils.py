import torch, pdb, random
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn as nn; import numpy as np; from typing import List

class LB_Init_Trainer:
    def __init__(self, args) -> None:
        from .lb_online_trainer_v7 import LB_Online_Trainer_V7

        trainer_type = getattr(args, 'trainer_type', None) # we do not have v1
        if trainer_type == 'v7':
            self.trainer_cls = LB_Online_Trainer_V7
        else:
            raise NotImplementedError
        

        pretrained_model = "openai/clip-vit-base-patch32"
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
        pretrained_model = "openai/clip-vit-base-patch32"
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
