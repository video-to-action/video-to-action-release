import numpy as np
import torch, gym, imageio, os
import os.path as osp

def center_crop_np(images: np.ndarray, size: tuple):
    """
    Center crops images in a batch to (128, 128).
    
    Parameters:
    - images: NumPy array of shape (B, H, W, 3).
    - size: a tuple of int (h, w), e.g., (128, 128)
    
    Returns:
    - A numpy of the cropped images with shape (B, 3, 128, 128).
    """
    
    B, H, W, _ = images.shape
    # start_h = H//2 - (128//2)
    # start_w = W//2 - (128//2)
    start_h = H//2 - (size[0]//2)
    start_w = W//2 - (size[1]//2)

    cropped = images[:, start_h:start_h+size[0], start_w:start_w+size[1], :]
    
    return cropped

def img_np_toTensor(images: np.ndarray, device=None):
    '''
    images: B, H, W, C
    returns: tensor B C H W, norm to 0-1
    '''
    assert images.dtype == np.uint8
    ## NOTE: copy is necessary for iThor env, but no need in MW or LB, can improve
    images = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float()
    if device is not None:
        images = images.to(device)
    return images / 255.0

def img_to_n1p1(images: torch.Tensor):
    '''assert already [0,1] normalize to [-1, 1]'''
    return images * 2 - 1
    
    

def train_img_preproc_np_Feb13(images, size, device):
    pass


def imgs_preproc_simple_v1(imgs, crop_size):
    '''simplest preprocessing
    center crop and to tensor and norm to [0,1]
    input B,H,W,C, 0-255, np uint8
    return 0-1, tensor float
    '''
    assert type(imgs) == np.ndarray and imgs.ndim == 4
    # imgs = np.array(imgs)
    imgs = center_crop_np(imgs, crop_size)
    imgs = img_np_toTensor(imgs) # in cpu, norm to [0,1]
    return imgs

def imgs_preproc_simple_noCrop_v1(imgs,): # crop_size):
    '''simplest preprocessing
    in Libero, we don't need crop
    input B,H,W,C, 0-255, np uint8
    return 0-1, tensor float
    '''
    assert type(imgs) == np.ndarray and imgs.ndim == 4
    # imgs = np.array(imgs)
    # imgs = center_crop_np(imgs, crop_size)
    imgs = img_np_toTensor(imgs) # in cpu, norm to [0,1]
    return imgs

def save_img_tr(img, root_dir, sub_dir, tk: str, c_name, env_idx:int,):
    tmp = osp.join( root_dir, sub_dir ); 
    os.makedirs(tmp, exist_ok=True)
    tmp = osp.join(tmp, f"{tk.replace(' ', '-')}-{c_name}-{env_idx}.png")
    imageio.imsave(tmp, img)
    print(f'[Save png] to {tmp}')


def save_gif_tr( imgs, root_dir, sub_dir, tk: str, c_name, env_idx:int ):
    assert imgs.dtype == np.uint8 and imgs.ndim == 4
    tmp = osp.join( root_dir, sub_dir ); 
    os.makedirs(tmp, exist_ok=True)
    tmp = osp.join(tmp, f"{tk.replace(' ', '-')}-{c_name}-{env_idx}.gif")
    imageio.mimsave(tmp, imgs, duration=0.5, )
    print(f'[Save gif] to {tmp}')


