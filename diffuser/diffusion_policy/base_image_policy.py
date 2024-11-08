from typing import Dict
import torch
import torch.nn as nn
from diffuser.diffusion_policy.common.module_attr_mixin import ModuleAttrMixin
from diffuser.diffusion_policy.normalizer import ConstNormalizerGroup

class BaseImagePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: ConstNormalizerGroup):
        raise NotImplementedError()
