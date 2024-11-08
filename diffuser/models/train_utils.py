import torch, pdb, random
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn as nn; import numpy as np; from typing import List
from diffuser.utils.eval_utils import print_color        


def freeze_model(model: nn.Module):
    assert isinstance(model, nn.Module)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

def freeze_trainer(trainer):
    print_color(f'[freeze_trainer] {type(trainer)}')
    for k, v in trainer.__dict__.items():
        if isinstance(v, torch.nn.Module):
            freeze_model(v)
            print(f'k: {k}, v {v.training}')

# ------------------------------------------------
# ---------- Runtime Helper Functions ------------

def rand_switch_cls_free(video_model, g_w,  cls_free_prob):
    # Trainer, float, float
    if random.random() < cls_free_prob:
        video_model.model.guidance_weight = g_w
        video_model.ema.ema_model.guidance_weight = g_w
    else:
        video_model.model.guidance_weight = 0
        video_model.ema.ema_model.guidance_weight = 0

def identity_np(t):
    assert isinstance(t, np.ndarray)
    return t

def identity_tensor(t):
    assert torch.is_tensor(t)
    return t

def merge_batch(*args, imgs_preproc_fn=identity_np):
    ''' 
    Used in training to merge one rand batch and one video batch
    args: a list of tuple( ..., ... )
    we'd better do the crop inside, because the original resolution might be high,
    and make img proc slow when batch size is large
    '''
    imgs_start, imgs_goal, acts_gt, tasks_str_input, srb_info = [], [], [], [], []
    for i_a, dp in enumerate(args):
        # imgs_start, imgs_goal, acts_gt, tasks_str_input, srb_info 
        st = imgs_preproc_fn(dp[0])
        gl = imgs_preproc_fn(dp[1])
        
        imgs_start.append(st) # a list of tensor [B,3,H,W]
        imgs_goal.append(gl) # a list of tensor [B,3,H,W]
        acts_gt.append(dp[2]) # a list of tensor [B,4]
        tasks_str_input.extend(dp[3])
        ## this dict should be merged
        srb_info.append(dp[4])
    
    # pdb.set_trace()
    ## batch level concatenate, e.g., [ [48, 3, 128, 128], [16, 3, 128, 128] ]
    if torch.is_tensor(imgs_start[0]):
        imgs_start = torch.cat(imgs_start, dim=0)
        imgs_goal = torch.cat(imgs_goal, dim=0)
    else:
        imgs_start = np.concatenate(imgs_start, axis=0)
        imgs_goal = np.concatenate(imgs_goal, axis=0)
    
    # after: (B1+B2, 4) or (B1+B2, H, 4)
    acts_gt = torch.cat(acts_gt, dim=0)
    assert len(tasks_str_input) == len(imgs_start)
    srb_info = merge_dicts(srb_info)
    
    return imgs_start, imgs_goal, acts_gt, tasks_str_input, srb_info






def merge_batch_grp(*args, imgs_preproc_fn=identity_np):
    ''' 
    Aug 3, 2024, for addutuibak camera vuew
    Used in training to merge one rand batch and one video batch
    args: a list of tuple( ..., ... )
    we'd better do the crop inside, because the original resolution might be high,
    and make img proc slow when batch size is large
    '''
    imgs_start, imgs_goal, acts_gt, tasks_str_input, srb_info = [], [], [], [], []
    imgs_grp = []
    for i_a, dp in enumerate(args):
        # imgs_start, imgs_goal, acts_gt, tasks_str_input, srb_info 
        st = imgs_preproc_fn(dp[0])
        gl = imgs_preproc_fn(dp[1])
        
        imgs_start.append(st) # a list of tensor [B,3,H,W]
        imgs_goal.append(gl) # a list of tensor [B,3,H,W]
        acts_gt.append(dp[2]) # a list of tensor [B,4]
        tasks_str_input.extend(dp[3])
        ## this dict should be merged
        srb_info.append(dp[4])
        imgs_grp.append(dp[5])

    
    # pdb.set_trace()
    ## batch level concatenate, e.g., [ [48, 3, 128, 128], [16, 3, 128, 128] ]
    if torch.is_tensor(imgs_start[0]):
        imgs_start = torch.cat(imgs_start, dim=0)
        imgs_goal = torch.cat(imgs_goal, dim=0)
        imgs_grp = torch.cat(imgs_grp, dim=0)

    else:
        imgs_start = np.concatenate(imgs_start, axis=0)
        imgs_goal = np.concatenate(imgs_goal, axis=0)
        imgs_grp = np.concatenate(imgs_grp, axis=0)

    
    # after: (B1+B2, 4) or (B1+B2, H, 4)
    acts_gt = torch.cat(acts_gt, dim=0)
    assert len(tasks_str_input) == len(imgs_start)
    assert len(tasks_str_input) == len(imgs_grp)
    srb_info = merge_dicts(srb_info)
    
    return imgs_start, imgs_goal, acts_gt, tasks_str_input, srb_info, imgs_grp







def merge_dicts(dict_list: List[dict]):
    '''
    assume all dicts have the same keys
    merge the element in each dict, and save to the large dict
    '''
    out = {}
    for k in dict_list[0]:
        out[k] = []
    
    for i_d in range(len(dict_list)):
        for k, v in dict_list[i_d].items():
            if not isinstance(v, np.ndarray):
                v = np.array(v)
            out[k].append(v)
    
    for k in dict_list[0]:
        out[k] = np.concatenate(out[k], axis=0)
        ## return to a normal list if it is string
        if out[k].dtype in ['U', 'S']:
            out[k] = out[k].tolist()
    
    return out



def weighted_uniform_sample(size:tuple, prob_0:float, low, mid, high):
    '''
    prob_0: prob that falls in the first interval
    low=-1, mid=0, high=1: int or np (1,)
    '''
    # Generate random numbers to decide the interval for each element
    # True -> low to mid; False -> mid to high
    interval_decisions = np.random.rand(*size) < prob_0
    
    # Generate uniform samples for each interval
    samples_negative = np.random.uniform(low=low, high=mid, size=size)
    samples_positive = np.random.uniform(low=mid, high=high, size=size)
    
    # Select from the appropriate interval for each element
    return np.where(interval_decisions, samples_negative, samples_positive)


