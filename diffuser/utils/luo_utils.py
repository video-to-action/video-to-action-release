import torch
import time
import gym
from mujoco_py import MjSimState
import numpy as np

def get_lr(optimizer):
    """Get the learning rate of current optimizer."""
    return optimizer.param_groups[0]['lr']

def get_tkname_vidmodel(task: str):
    '''get the text prompt for the video difusion (vd) model'''
    if 'v2' in task:
        return " ".join(task.split('-')[:-3])
    else:
        return " ".join(task.split('-'))


def batch_repeat_tensor(x: torch.Tensor, t, task_embed, n_rp: int):
    '''
    deepcopy tensor along batch dim for eval pipeline
    n_rp: num of repeat
    '''
    x_shape = [n_rp,] + [1] * (x.dim() - 1)
    x = x.repeat( *x_shape )
    
    t_shape = [n_rp,] + [1] * (t.dim() - 1)
    t = t.repeat( *t_shape ) # (B,)

    te_shape = [n_rp,] + [1] * (task_embed.dim() - 1)
    task_embed = task_embed.repeat( *te_shape ) # (B, 4, 512)

    return x, t, task_embed


class Timer:

	def __init__(self):
		self._start = time.time()

	def __call__(self, reset=True):
		now = time.time()
		diff = now - self._start
		if reset:
			self._start = now
		return diff
      

def load_environment(name, **kwargs):
    '''would be call several times: 2 in dataset, 1 in rendering'''
    import environment
    # traceback.print_stack()
    if type(name) != str:
        ## name is already an environment
        assert hasattr(name, 'name')
        return name # copy.deepcopy(name)
    # with suppress_output():
        # wrapped_env = gym.make(name)
    kwargs['name'] = name
    wrapped_env = gym.make(name, load_mazeEnv=False, **kwargs) # min_episode_distance=4.0
    
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    
    return env

def equal_sim_states(state_1: MjSimState, state_2):
    '''check if two states are identical'''
    # sometimes the simulation will make error 1e-10 level
    c1 = ( np.abs( state_1[0].qpos - state_2[0].qpos ) < 1e-6 ).all()
    c2 =  (state_1[1][0] == state_2[1][0]).all()
    c3 =  (state_1[1][1] == state_2[1][1]).all()
    return c1 and c2 and c3
    
def custom_sim_state(j_state: MjSimState, qpos, qvel):
    pass
     
def custom_sim_state_2(env, time=None, qpos=None, qvel=None, set_to_env=False):
    j_st, m_st = env.get_env_state()
    j_st = list(j_st)
    if time is not None:
        j_st[0] = time
    if qpos is not None:
        j_st[1] = qpos
    if qvel is not None:
        j_st[2] = qvel
    
    j_st = MjSimState(*j_st)

    if set_to_env:
        env.set_env_state( (j_st, m_st) )
    

    
     
     