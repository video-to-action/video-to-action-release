import os
from PIL import Image

def set_read_only(fname):
    os.chmod( fname, 0o444 )



def set_read_only_list(fnames):
    for fname in fnames:
        if 'luotest' not in fname:
            os.chmod( fname, 0o444 )


def load_an_img(img_path):
    '''return a PIL Image'''
    with open(img_path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")
    return img

import h5py, pdb
def load_an_img_hdf5_cal(h5_path, tk, ep_idx, key_name, fr_idx:int):
    '''return a PIL Image'''
    with h5py.File(h5_path, 'r') as hd_f:
        ## (200, 200, 3)
        img = hd_f[tk][ep_idx][key_name][ fr_idx ] ## a numpy array
        # pdb.set_trace()
        img = Image.fromarray(img)
        # pdb.set_trace()
        img = img.convert("RGB")
        # pdb.set_trace()
    
    ## return an uint8 [0,255] image
    return img

