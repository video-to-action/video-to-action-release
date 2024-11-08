import collections
import numpy as np
import torch
import pdb

DTYPE = torch.float
# DEVICE = 'cuda:0'
#-----------------------------------------------------------------------------#
#------------------------------ NEW Feb 2024 ---------------------------------#
def torch_stack(*args, dim):
	return (torch.stack(arg, dim=dim) for arg in args)

def to_device_tp(*args, device):
    ## args is a tuple
    return tuple(arg.to(device) for arg in args)

def to_torch_tp(*args, dtype=torch.float32, device='cpu'):
	return  ( to_torch(arg, dtype=dtype, device=device) for arg in args )


def number_by_ratio(num, ratio):
	'''Example: num=10, ratio=[0.2, 0.8] -> [2, 8]
	return a list of int
	'''
	ratio = np.array(ratio)
	assert np.isclose(sum(ratio), 1), "Ratios must sum to 1"
	out = (ratio * num)
	assert np.isclose(out.sum(), num)
	out = np.round(out).astype(np.int32).tolist()
	# print(out)
	return out



#-----------------------------------------------------------------------------#
#------------------------------ numpy <--> torch -----------------------------#
#-----------------------------------------------------------------------------#

def to_np(x):
	if torch.is_tensor(x):
		x = x.detach().cpu().numpy()
	return x




def to_torch(x, dtype, device):
	dtype = dtype or DTYPE
	if type(x) is dict:
		return {k: to_torch(v, dtype, device) for k, v in x.items()}
	elif torch.is_tensor(x):
		return x.to(device).type(dtype)

	return torch.tensor(x, dtype=dtype, device=device)




def batchify_tp(*args, device='cpu'):
	'''
		convert a single dataset item to a batch suitable for passing to a model by
			1) converting np arrays to torch tensors and
			2) and ensuring that everything has a batch dimension
	'''
	return tuple(arg[None,].to(device) for arg in args)


def apply_dict(fn, d, *args, **kwargs):
	"""update every value in d:dict with function fn"""
	return {
		k: fn(v, *args, **kwargs)
		for k, v in d.items()
	}




def _to_str(num):
	if num >= 1e6:
		return f'{(num/1e6):.2f} M'
	else:
		return f'{(num/1e3):.2f} k'

#-----------------------------------------------------------------------------#
#----------------------------- parameter counting ----------------------------#
#-----------------------------------------------------------------------------#

def param_to_module(param):
	module_name = param[::-1].split('.', maxsplit=1)[-1][::-1]
	return module_name

def report_parameters(model, topk=10):
	counts = {k: p.numel() for k, p in model.named_parameters()}
	n_parameters = sum(counts.values())
	print(f'[ utils/arrays ] Total parameters: {_to_str(n_parameters)}')

	modules = dict(model.named_modules())
	sorted_keys = sorted(counts, key=lambda x: -counts[x])
	max_length = max([len(k) for k in sorted_keys])
	for i in range(topk):
		key = sorted_keys[i]
		count = counts[key]
		module = param_to_module(key)
		print(' '*8, f'{key:10}: {_to_str(count)} | {modules[module]}')

	remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
	print(' '*8, f'... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters')
	return n_parameters
