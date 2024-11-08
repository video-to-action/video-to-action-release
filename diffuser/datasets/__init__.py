import numpy as np

## verified
MW_SAWYER_ACTION_MIN = np.array([-1., -1., -1., -1.], dtype=np.float32)
MW_SAWYER_ACTION_MAX = np.array([1., 1., 1., 1.], dtype=np.float32)
MW_SAWYER_ACTION_LEN = 4

# image_minmax_01 = [[0,0,0], [1,1,1]]
IMAGE_MINMAX_01 = ( np.array([0,0,0], dtype=np.float32), np.array([1,1,1], dtype=np.float32) )
def image_minmax_01_f():
    return (*IMAGE_MINMAX_01, [1,3,1,1])

MW_SAWYER_ACTION_MINMAX = (MW_SAWYER_ACTION_MIN, MW_SAWYER_ACTION_MAX)
def mw_sawyer_action_minmax_f():
    return (*MW_SAWYER_ACTION_MINMAX, [1, 4])

# MW_SAWYER_AGENT_POS_MINMAX = ()

## For Libero
## 1. A Normal Range
LB_ACTION_MIN = np.array([-1., -1., -1., -1., -1., -1., -1.], dtype=np.float32)
LB_ACTION_MAX = np.array([1., 1., 1., 1., 1., 1., 1.,], dtype=np.float32)
LB_ACTION_LEN = 7
assert len(LB_ACTION_MIN) == len(LB_ACTION_MAX) == LB_ACTION_LEN
LB_ACTION_MINMAX = (LB_ACTION_MIN, LB_ACTION_MAX)
def lb_action_minmax_f():
    return (*LB_ACTION_MINMAX, [1, 7])



## 2. orn in range(-0.1, 0.1)
LB_ACTION_MIN_orn01 = np.array([-1.,]*3 + [-0.1]*3 + [-1.,], dtype=np.float32)
LB_ACTION_MAX_orn01 = np.array([1.,]*3 + [0.1]*3 + [1.,], dtype=np.float32)
assert len(LB_ACTION_MIN_orn01) == len(LB_ACTION_MAX_orn01) == LB_ACTION_LEN

LB_ACTION_MINMAX_orn01 = (LB_ACTION_MIN_orn01, LB_ACTION_MAX_orn01)

def lb_action_minmax_orn01_f():
    return (*LB_ACTION_MINMAX_orn01, [1, 7])

## Text, the min and max is not ture, just a placeholder
Task_Embed_MIN = np.array([0.,]*512, dtype=np.float32)
Task_Embed_MAX = np.array([1.,]*512, dtype=np.float32)
def tk_emb_minmax_f():
    return (Task_Embed_MIN, Task_Embed_MAX, [1, 512])


## ---------------------------------
## For Thor
## discrete
Thor_ACTION_MIN_Dim4 = np.array([-1., -1., -1., -1.], dtype=np.float32)
Thor_ACTION_MAX_Dim4 = np.array([1., 1., 1., 1.], dtype=np.float32)
Thor_ACTION_LEN_Dim4 = 4
assert len(Thor_ACTION_MIN_Dim4) == len(Thor_ACTION_MAX_Dim4) == Thor_ACTION_LEN_Dim4
Thor_ACTION_MINMAX_Dim4 = (Thor_ACTION_MIN_Dim4, Thor_ACTION_MAX_Dim4)

def thor_action_minmax_dim4_f():
    return (*Thor_ACTION_MINMAX_Dim4, [1, 4])


## Sep 4 For Calvin
CAL_ACTION_MIN = np.array([-1., -1., -1., -1., -1., -1., -1.], dtype=np.float32)
CAL_ACTION_MAX = np.array([1., 1., 1., 1., 1., 1., 1.,], dtype=np.float32)
CAL_ACTION_LEN = 7
assert len(CAL_ACTION_MIN) == len(CAL_ACTION_MAX) == CAL_ACTION_LEN
CAL_ACTION_MINMAX = (CAL_ACTION_MIN, CAL_ACTION_MAX)
def cal_action_minmax_f():
    return (*CAL_ACTION_MINMAX, [1, 7])

## v2
CAL_abs_ACTION_MIN = np.array([-0.20, -0.50,  0.3, -3.15, -0.50, -3.15, -1.], dtype=np.float32) - 0.01
CAL_abs_ACTION_MAX = np.array([0.36 ,  0.12,  0.70,  3.15,  0.30,  3.15, 1.], dtype=np.float32) + 0.01
CAL_abs_ACTION_MINMAX = (CAL_abs_ACTION_MIN, CAL_abs_ACTION_MAX)

def cal_abs_action_minmax_f():
    return (*CAL_abs_ACTION_MINMAX, [1, 7])
