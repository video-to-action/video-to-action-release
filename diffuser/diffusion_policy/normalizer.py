import numpy as np
import scipy.interpolate as interpolate
import torch, pdb


class ConstNormalizerGroup:
    """
    divide the data by a given constant to normalize data to range [-1, 1]
    """
    def __init__(self, normalizer, shape_meta, n_obs_steps, use_tensor=True):

        if type(normalizer) == str:
            normalizer = eval(normalizer)

        self.normalizers = {}

        norm_const_dict = { **shape_meta['obs'] }
        norm_const_dict['action'] = shape_meta['action']

        print('norm_const_dict:', norm_const_dict.keys())
        print(f'normalizer: {normalizer}')

        ## loop through obs_1, goal_1, ..., action
        ## create a normalizer for each key, LimitsConstNormalizer by default
        for key, val in norm_const_dict.items(): # val is a dict
            # print('key:', key); pdb.set_trace()
            ## direcly use the given value to normalize
            assert 'LimitsConstNormalizer' in str(normalizer)
            # [0] min, [1] max, [2] a shape
            mms_tmp = val['minmax_shape']
            if use_tensor: # v just stands for value
                v_min, v_max = torch.from_numpy(mms_tmp[0]), torch.from_numpy(mms_tmp[1])
            else:
                v_min, v_max = mms_tmp[0], mms_tmp[1] # numpy
            
            ## shape with T
            if key == 'action':
                n_steps = 1
            else:
                n_steps = 1 # n_obs_steps
            v_shape = (mms_tmp[2][0], n_steps, *mms_tmp[2][1:])

            self.normalizers[key] = normalizer( (v_min, v_max), v_shape)
            
            print(f'key {key}, min {self.normalizers[key].mins.shape}') # min is 1D


    def __repr__(self):
        string = ''
        for key, normalizer in self.normalizers.items():
            string += f'{key}: {normalizer}]\n'
        return string

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def normalize_d(self, obs_dict):
        '''
        normalize a dict of observations
        corner_st, corner2_st, corner3_st
        corner_gl, corner2_gl, corner3_gl
        agent_pos
        '''
        # assert len(obs_dict) == len(self.normalizers)
        result = dict()
        for key, value in obs_dict.items():
            result[key] = self.normalize(obs_dict[key], key)

        return result


    def __getitem__(self, key):
        return self.normalizers[key]

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)
    
    def to_device(self, *args, **kwargs):
        for key, val in self.normalizers.items():
            val.mins = val.mins.to(*args, **kwargs)
            val.maxs = val.maxs.to(*args, **kwargs)
    @property
    def device(self):
        m_tmp = next(iter(self.normalizers.values())).mins
        if torch.is_tensor(m_tmp):
            return m_tmp.device
        else:
            return None

class ConstNormalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
        NOTE that the minmax is the range of the input data, which is used to norm to [-1,1]
    '''

    def __init__(self, min_max, in_shape):
        ''' we need to support given custom min max value.
        min_max: a tuple of np1d (min, max)
        in_shape: input shape (B,D) or (B,3,H,W)
        '''
        
        self.mins = min_max[0]
        self.maxs = min_max[1]

        self.mins = self.mins.reshape(*in_shape)
        self.maxs = self.maxs.reshape(*in_shape)

        # print(f'self.mins: {self.mins}')
        # assert self.mins.shape == X.min(axis=0).shape
        # assert self.mins.ndim == 1


    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            # f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
            f'''{self.mins}\n    +: {self.maxs}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()



class LimitsConstNormalizer(ConstNormalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''

    def normalize(self, x):
        ## NOTE mins and maxs are computed on axis=0 only
        ## [B,horizon,correct_dim]
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=0): # 1e-4, might be out of limit
        '''
            x : [ -1, 1 ]
        '''
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            if torch.is_tensor(x):
                x = torch.clamp(x, -1, 1)
            else:
                x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins