from gym.envs.registration import register
import numpy as np
import os

if os.getenv('CONDA_DEFAULT_ENV').split('/')[-1] in ['v2a_libero_release',]:
    from environment.libero.init_libero import register_libero
    register_libero()
