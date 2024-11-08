import math, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pdb

class WeightedStateLoss(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'a0_loss': weighted_loss}
    

class WeightedStateL2(WeightedStateLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


def get_no_dash_tasks_str(tasks_str): 
    '''given a list task str, prepare it for video diffusion input,
      because no '-' is allowed'''
    tks_new = []
    for tk in tasks_str:
        tmp_s = tk.split('-')
        assert len(tmp_s) <= 3
        tks_new.append( " ".join(tmp_s) )
    # pdb.set_trace()
    return tks_new
        

def get_no_underscore_tasks_str(tasks_str): 
    '''given a list task str, prepare it for video diffusion input,
      because no '-' is allowed'''
    tks_new = []
    for tk in tasks_str:
        tmp_s = tk.split('_')
        assert len(tmp_s) <= 3
        tks_new.append( " ".join(tmp_s) )
    
    return tks_new