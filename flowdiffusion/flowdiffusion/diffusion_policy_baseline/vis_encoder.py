import torchvision
from torch import nn
import torch
import torchvision.transforms as T
import numpy as np

def replace_submodules(
        root_module: nn.Module, 
        predicate=lambda x: isinstance(x, nn.BatchNorm2d), 
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//16, 
            num_channels=x.num_features)) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

class SpatialSoftmaxPooling(nn.Module):
    def __init__(self, w=128, h=128):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.w = w/16
        self.h = h/16

    def forward(self, x):
        b, c, h, w = x.shape 
        x = self.softmax(x.view(b, c, -1)).view(b, c, h, w)
        # Get the expectance h w position
        exp_x = torch.sum(x * torch.arange(w)[None, None, None, :].float().to(x.device)/self.w, dim=[-1, -2])
        exp_y = torch.sum(x * torch.arange(h)[None, None, :, None].float().to(x.device)/self.h, dim=[-1, -2])
        x = torch.cat([exp_x, exp_y], dim=-1)
        return x

class ResNet18Encoder(nn.Module):
    def __init__(self, input_resolution=(128, 128), output_dim=512):
        super().__init__()
        model = torchvision.models.resnet18(pretrained=False)
        model = replace_submodules(model)
        
        model.avgpool = SpatialSoftmaxPooling(w=input_resolution[0], h=input_resolution[1])
        model.fc = nn.Linear(1024, output_dim)
        self.model = model
    
    def forward(self, x):
        return self.model(x)

