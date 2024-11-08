import numpy as np
import torch, pdb, contextlib, os, json
import torch.nn.functional as F
import diffuser.utils as utils
from colorama import Fore
import einops, imageio, math
from datetime import datetime
from torchvision import transforms as T
from PIL import Image
from typing import List
from einops import rearrange
import os.path as osp


## ----------------------------------
## ------------ Save ----------------

def save_img(save_path: str, img: np.ndarray):
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    imageio.imsave(save_path, img)
    print(f'[save_img] {save_path}')

def save_imgs_1(imgs, full_paths):
    for i_m, img in enumerate(imgs):
        p_dir = osp.dirname(full_paths[i_m])
        os.makedirs(p_dir, exist_ok=True)
        imageio.imsave(full_paths[i_m], img)
    
    print(f'save {len(imgs)} pngs to:', p_dir)

def save_gif(imgs, root_dir, fname, dr=0.5, se_dr=1.5):
    '''
    dr: duration
    se_dr: duration for the start and end
    '''
    if type(imgs) == np.ndarray:
        assert imgs.ndim == 4
        ## float 0-1 to uint8
        if imgs.dtype == np.float32:
            assert imgs.min() >= 0 and imgs.max() <= 1
            imgs = (imgs * 255).astype(np.uint8)
        else:
            assert imgs.dtype == np.uint8
    else:
        assert type(imgs) == list
        assert imgs[0].ndim == 3
        ## float 0-1 to uint8
        if imgs[0].dtype == np.float32:
            assert imgs[0].min() >= 0 and imgs[0].max() <= 1
            imgs = [(img * 255).astype(np.uint8) for img in imgs]
        else:
            assert imgs[0].dtype == np.uint8


    os.makedirs(root_dir, exist_ok=True)
    tmp = osp.join(root_dir, f'{fname}')
    ds = [dr,] * len(imgs)
    ds[0] = se_dr; ds[-1] = se_dr
    ### newer version the ds is in milli-second
    # if type(dr) != np.ndarray and dr < 10:
        # ds = (np.array(ds) * 1000).tolist()
    # pdb.set_trace()
    imageio.mimsave(tmp, imgs, duration=ds)
    # pdb.set_trace()
    print(f'[save_gif] to {tmp}')
    
    return tmp

def save_fig(fig, root_dir, fname):
    '''direct save a plt fig to fname'''
    
    os.makedirs(root_dir, exist_ok=True)
    tmp = osp.join(root_dir, f'{fname}')
    img = plot2img(fig,)
    imageio.imsave(tmp, img)
    print(f'[save_fig] to {tmp}')



def save_json(j_data: dict, full_path):
    with open(full_path, "w") as f:
        json.dump(j_data, f, indent=4)
    print(f'[save_json] {full_path}')



# -------------------------------------------
# --------- Visualization Gadgets -----------
import matplotlib.pyplot as plt

def plt_img(img, dpi=100):
    plt.clf(); plt.figure(dpi=dpi); plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if img.shape[0] == 1 and img.ndim == 4:
        img = img[0]
    if img.shape[0] in [1, 3]:
        img = img.transpose(1, 2, 0)
    plt.imshow(img)
    plt.margins(x=0)
    plt.margins(y=0)
    plt.show()

def plt_imgs_grid(imgs, dpi=80, n_rows=1, texts=None, caption=None, show=False, max_n_cols=6):
    """
    Plots a list of images in a grid.
    
    Parameters:
    - images: List[np, tensor] of image arrays.
    - n_rows: Number of rows in the grid.
    - texts: corresponds to per subplot
    """
    # Calculate the number of columns, rounding up
    n_cols = math.ceil(len(imgs) / n_rows)
    if n_cols > max_n_cols:
        n_cols = max_n_cols
        n_rows = math.ceil(len(imgs) / n_cols)
    # Set up the figure size dynamically based on the number of images
    plt.clf(); sz = 3
    fig = plt.figure(figsize=(n_cols*sz, n_rows*sz), dpi=dpi)
    if texts is None:
        texts = list(range(len(imgs)))
    
    for i, img in enumerate(imgs, 0):
        plt.subplot(n_rows, n_cols, i+1)
        if img.shape[0] in [1, 3]: # [3, H, W]
            ## might be [0-1], {0,255}
            if torch.is_tensor(img):
                img = img.permute(1, 2, 0).cpu()
            else:
                img = img.transpose(1, 2, 0)
        plt.imshow(img)
        if texts is not None:
            plt.title(texts[i])
        plt.axis('off')
    if caption is not None:
        # Provide an overall caption/title for the figure
        plt.suptitle(caption, fontsize=28)
    plt.tight_layout()
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot2img(fig, remove_margins=False):
    '''
    seems to be slow, not efficient
    '''
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349

    ## fig.tight_layout(pad=0.1)
    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))


def cat_2_figs(fig1, fig2):
    img1 = utils.plot2img(fig1)
    img2 = utils.plot2img(fig2)
    try:
        img12 = np.concatenate([img1, img2], axis=0)
    except:
        img12 = img1
    return img12

def cat_n_figs(figs: list):
    imgs = []
    for fig in figs:
        # img1 = utils.plot2img(fig1)
        # img2 = utils.plot2img(fig2)
        imgs.append(utils.plot2img(fig))
    try:
        img12 = np.concatenate(imgs, axis=0)
    except:
        img12 = imgs[0]
    return img12

def img_tensor_to_display(img):
    '''convert a 3HW tensor float 0-1 img to HW3 np uint 8 img '''
    assert img.ndim == 3 and torch.is_tensor(img)
    return (img.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)

def imgs_tensor_to_display(imgs):
    return [img_tensor_to_display(img) for img in imgs]


## -------------------------------------
## -------------------------------------




def print_color(s, *args, c='r'):
    if c == 'r':
        # print(Fore.RED + s + Fore.RESET)
        print(Fore.RED, end='')
        print(s, *args, Fore.RESET)
    elif c == 'b':
        # print(Fore.BLUE + s + Fore.RESET)
        print(Fore.BLUE, end='')
        print(s, *args, Fore.RESET)
    elif c == 'y':
        # print(Fore.YELLOW + s + Fore.RESET)
        print(Fore.YELLOW, end='')
        print(s, *args, Fore.RESET)
    else:
        # print(Fore.CYAN + s + Fore.RESET)
        print(Fore.CYAN, end='')
        print(s, *args, Fore.RESET)
    
    

def get_time():
    return datetime.now().strftime("%y%m%d-%H%M%S")

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as nullfile:
        with contextlib.redirect_stdout(nullfile):
            yield



def gif_to_numpy_array(gif_path):
    # Open the GIF file
    with Image.open(gif_path) as gif:
        frames = []
        for frame_index in range(gif.n_frames):
            gif.seek(frame_index)  # Move to the next frame
            frame = gif.convert("RGBA")  # Convert the frame to RGBA (4-channel image)
            frame_array = np.array(frame)  # Convert the frame to a NumPy array
            frames.append(frame_array)  # Append to the list of frames
        
        # Stack all frames into a NumPy array (shape: [num_frames, height, width, channels])
        gif_array = np.stack(frames, axis=0)
    
    return gif_array


### -----------------------------------------
### ------------- MP4 -----------------------
### -----------------------------------------

def save_imgs_to_mp4(imgs, save_path, fps=24, n_repeat_first=0, n_rep_list=None):
    """
    Saves a list of NumPy imgs to an MP4 video file using imageio.

    Parameters:
    ----------
    imgs : list of np.ndarray
        List of imgs to be saved as video frames. Each image should be a NumPy array.
    save_path : str
        The filename for the output video file (e.g., 'output_video.mp4').
    fps : int, optional
        Frames per second for the output video. Default is 24.

    Returns:
    -------
    None

    Notes:
    -----
    - All imgs must have the same dimensions and number of channels.
    - imgs should be in uint8 format. If not, they will be converted.
    - If imgs are grayscale (2D arrays), they will be converted to RGB.
    """
    import imageio
    import numpy as np
    if torch.is_tensor(imgs):
        assert imgs.ndim == 4
        if imgs.shape[1] == 3:
            imgs = imgs.permute(0,2,3,1).numpy()
    if n_repeat_first > 0:
        print(f'before: {imgs.shape=}')
        imgs = np.concatenate(   [imgs[0:1],] * n_repeat_first + [ imgs, ], axis=0  )
        print(f'after: {imgs.shape=}')
    elif n_rep_list is not None:
        num_fr = len(imgs)
        assert len(n_rep_list) == num_fr
        imgs_ori = imgs
        imgs = []
        ## repeat the corresponding frame n times
        for i_im in range(num_fr):
            imgs.extend( [imgs_ori[i_im],] * n_rep_list[i_im] )

    # Validate inputs
    # if not imgs:
        # raise ValueError("The list of images is empty.")

    if not isinstance(save_path, str):
        raise TypeError("Output filename must be a string.")

    ## Check ffmpeg availability
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with imageio.get_writer(save_path, fps=fps) as writer:
        for idx, img in enumerate(imgs):
            # Convert image to uint8 if necessary
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 1) if img.dtype in [np.float32, np.float64] else img
                img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img.astype(np.uint8)
            
            # Convert grayscale to RGB
            if img.ndim == 2:
                img = np.stack((img,)*3, axis=-1)
            
            # Validate image dimensions
            if img.ndim != 3 or img.shape[2] not in [1, 3, 4]:
                raise ValueError(f"Invalid image shape at index {idx}: expected 2D or 3D array with 1, 3, or 4 channels.")

            # Append frame to video
            writer.append_data(img)

    print(f'[save_imgs_to_mp4] to {save_path}')
    return 


def load_video_frames(filename):
    """
    Loads frames from a video file (GIF or MP4) into a list of NumPy arrays.

    Parameters:
    ----------
    filename : str
        Path to the video file.

    Returns:
    -------
    frames : list of np.ndarray
        List containing the video frames as NumPy arrays.
    """
    reader = imageio.get_reader(filename)
    frames = []
    for frame in reader:
        frames.append(frame)
    reader.close()
    return frames
