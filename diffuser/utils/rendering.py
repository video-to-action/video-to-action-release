from diffuser.utils.eval_utils import plt_imgs_grid
from matplotlib.backends.backend_agg import FigureCanvasAgg
import diffuser.utils as utils
from diffuser.utils import plot2img
import numpy as np
import imageio

class MW_Renderer:
    def __init__(self, env_list) -> None:
        """
        This class is probably not used.
        """
        self.env_list = env_list
        self.crop_size = (128, 128)
        # self.dpi = 50
        pass

    def render_states_seq(self, tk: str, states, cam_name, resol, 
                          savepath=None, env_idx='', env=None, imgs_fn=None):
        '''
        given a list of states of a specific task, render them
        can be used to visualize the states in the replay buffer
        Params:
        - env: a gym env instance, if given, ignore the env_list
        '''
        if env is None:
            env = self.env_list.env_list[tk]
            ## useless
            if getattr(self.env_list, 'version') == 'v2':
                env = env[env_idx]
        ori_state = env.get_env_state()
        imgs = []
        
        for i_s, state in enumerate(states):
            env.set_env_state(state)
            img = env.render(camera_name=cam_name, resolution=resol)
            
            imgs.append( img )
        
        imgs = np.array(imgs)
        if imgs_fn is not None:
            imgs = imgs_fn(imgs, self.crop_size)
        ## reset
        env.set_env_state(ori_state)
        
        cpt = f'{tk}-{cam_name} {env_idx}'.strip()
        fig = plt_imgs_grid(imgs, caption=cpt)
        img_r = plot2img(fig, False)
        if savepath is not None:
            imageio.imsave(savepath, img_r)
            print(f'Saved states seq to: {savepath}')
        return img_r
    
    def render_startgoals(self, imgs_start, imgs_goal, tks, env_idxs, savepath=None):
        '''can render a batch of start&goal pairs in one image'''
        # if env is None:
            # env = self.env_list.env_list[tk]
        tks = len_align_Empty_str(tks)
        env_idxs = len_align_Empty_str(env_idxs)

        imgs = []
        texts = []
        for i_t, tk in enumerate(tks):
            imgs.append( imgs_start[i_t] )
            imgs.append( imgs_goal[i_t] )
            texts.append( f'{tk} {env_idxs[i_t]} start'.strip() )
            texts.append( f'{tk} {env_idxs[i_t]} goal'.strip() )
        
        fig = plt_imgs_grid(imgs)
        img_r = plot2img(fig, False)
        if savepath is not None:
            imageio.imsave(savepath, img_r)
            print(f'Saved start&goal pairs to: {savepath}')
        return img_r
    
    def render_img_grid(self, imgs, caption='', savepath=None):
        '''can render a batch of start&goal pairs in one image'''

        texts = [i for i in range(len(imgs))]
        fig = plt_imgs_grid(imgs, texts=texts, caption=caption)
        img_r = plot2img(fig, False)
        if savepath is not None:
            imageio.imsave(savepath, img_r)
            print(f'Saved start&goal pairs to: {savepath}')
        return img_r



def len_align_Empty_str(aa, tks):
    if tks is None:
        return ['',] * len(aa)
    else:
        return tks

