import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

def lb_rand_sample_1_ep(env_u: OffScreenRenderEnv, rs_cfg: dict):
    '''
    From libero/notebook/luotest_90_taskorder.ipynb
    random sample one episode in the given libero env
    '''
    # env_u = env_list[0]
    obs = env_u.reset()

    # ## --------------------------
    # ## ----- hyper params -------
    # # orn_sample_range = (-0.05, 0.05)
    # ## this serve as mocap!! limit the robot actions in the given range!!
    # # x_limit = (-0.32, 0.32) ## facing cameras
    # x_limit = (-0.30, 0.25) ## facing cameras
    # # y_limit = (-0.32, 0.32) ## left / right
    # # y_limit = (-0.36, 0.36) ## left / right
    # y_limit = (-0.38, 0.38) ## left / right
    # # z_limit = (0.0, 0.75)
    # z_limit = (0.0, 0.80)
    # is_stop_when_out = True
    # rand_act_noise_std = 0.003
    # act_min_np = - np.ones(shape=(7,))
    # act_max_np = np.ones(shape=(7,))
    # rr_tmp = [[-0.981,-0.98], [0.98, 0.981]]
    # rand_act_gripper_ranges = np.array(rr_tmp)
    # num_gripper_ranges = len(rand_act_gripper_ranges)
    # rand_ep_len = 120 ## len of actions
    # rand_act_full_len = 24
    # orn_sample_range = (-0.1, 0.1)

    # ## --------------------------
    # ## --------------------------

    ## --------------------------
    ## ----- hyper params -------
    ## this serve as mocap!! limit the robot actions in the given range!!
    x_limit = rs_cfg['x_limit'] # (-0.30, 0.25) ## facing cameras
    y_limit = rs_cfg['y_limit'] # (-0.38, 0.38) ## left / right
    z_limit = rs_cfg['z_limit'] # (0.0, 0.80)

    is_stop_when_out = rs_cfg['is_stop_when_out'] # True
    rand_act_noise_std = rs_cfg['rand_act_noise_std']  # 0.003
    rand_act_noise_std_orn = rs_cfg['rand_act_noise_std_orn']  # 0.003
    act_min_np = np.array(rs_cfg['act_min_np'], dtype=np.float32) # - np.ones(shape=(7,))
    act_max_np = np.array(rs_cfg['act_max_np'], dtype=np.float32) # np.ones(shape=(7,))


    # rr_tmp = [[-0.981,-0.98], [0.98, 0.981]]
    rr_tmp = rs_cfg['r_a_g_ranges']
    rand_act_gripper_ranges = np.array(rr_tmp)
    num_gripper_ranges = len(rand_act_gripper_ranges)
    
    rand_ep_len = rs_cfg['rand_ep_len'] # 120 ## len of actions
    rand_act_full_len =  rs_cfg['rand_act_full_len'] # 24
    orn_sample_range = rs_cfg['orn_sample_range'] # (-0.1, 0.1)

    assert np.isclose( act_min_np[:3], - np.ones(shape=(3,)) ).all(), 'the below sampling only support this'
    assert np.isclose( act_max_np[:3], np.ones(shape=(3,)) ).all()
    assert np.isclose( act_min_np[3:6], np.full(shape=3, fill_value=orn_sample_range[0]) ).all(), 'the below sampling only support this'
    assert np.isclose( act_max_np[3:6], np.full(shape=3, fill_value=orn_sample_range[1]) ).all()
    
    assert np.isclose( act_min_np[6], -1 )
    assert np.isclose( act_max_np[6], 1 )
    # breakpoint()
    ## --------------------------
    ## --------------------------





    cur_ee_pos = obs['robot0_eef_pos']
    ee_poses_ep = [cur_ee_pos,]
    cur_img = obs['agentview_image']
    imgs_ep = [cur_img,]
    acts_ep = []

    # for _ in range(15):
    while len(acts_ep) < rand_ep_len:
        ## 1. pure random
        # act_u = np.random.uniform(low=-1, high=1, size=(3,)).tolist()  + [0.00] * 3 + [1.0]
        ## 2. heuristic
        x_cur, y_cur, z_cur = ee_poses_ep[-1]

        ## sample X
        if x_cur < x_limit[0]: # < -0.32
            x_rd = np.random.uniform(low=0, high=1, size=(1,))
        elif x_cur > x_limit[1]: # > 0.32
            x_rd = np.random.uniform(low=-1, high=0, size=(1,))
        else:
            x_rd = np.random.uniform(low=-1, high=1, size=(1,))
        
        ## sample Y
        if y_cur < y_limit[0]:
            y_rd = np.random.uniform(low=0, high=1, size=(1,))
        elif y_cur > y_limit[1]:
            y_rd = np.random.uniform(low=-1, high=0, size=(1,))
        else:
            y_rd = np.random.uniform(low=-1, high=1, size=(1,))
        
        ## sample Z
        if z_cur < z_limit[0]:
            assert False, 'impossible'
        elif z_cur > z_limit[1]:
            z_rd = np.random.uniform(low=-1, high=0, size=(1,))
        else:
            z_rd = np.random.uniform(low=-1, high=1, size=(1,))


        orn_rd = np.random.uniform(low=orn_sample_range[0], high=orn_sample_range[1], size=(3,))
        
        ### For the gripper
        if True:
            r_idx = np.random.randint(num_gripper_ranges, size=(1,)).item()
            sel_range = rand_act_gripper_ranges[r_idx] # [low, high]
            grp_rd = np.random.uniform(low=sel_range[0], high=sel_range[1], size=(1,))
            print('gripper:', r_idx, sel_range, grp_rd)

        act_u = np.concatenate([x_rd, y_rd, z_rd, orn_rd, grp_rd], axis=0)
        assert act_u.shape == (7,)

        print(f'{len(ee_poses_ep)}', 'act_u', act_u[:])
        
        for i_step in range(rand_act_full_len):
            # noise = np.random.normal(loc=0, scale=rand_act_noise_std, 
                                    # size=act_u.shape).astype(np.float32)
            noise_1 = np.random.normal(loc=0, scale=rand_act_noise_std, 
                                    size=(4,)).astype(np.float32)
            noise_2 = np.random.normal(loc=0, scale=rand_act_noise_std_orn, 
                                    size=(3,)).astype(np.float32)
            noise = np.concatenate([noise_1[:3], noise_2, noise_1[3:4]], axis=0)
            assert noise.shape == act_u.shape

            act_u_noisy = act_u + noise
            act_u_noisy = np.clip(act_u_noisy, a_min=act_min_np, a_max=act_max_np)

            obs,_,_,_ = env_u.step(act_u_noisy)
            
            acts_ep.append(act_u_noisy)
            imgs_ep.append(obs['agentview_image'])
            
            cur_ee_pos = obs['robot0_eef_pos']
            ee_poses_ep.append(obs['robot0_eef_pos'])
            
            if is_stop_when_out:
                x_cur, y_cur, z_cur = cur_ee_pos
                is_x_out = x_cur < x_limit[0] or x_cur > x_limit[1]
                is_y_out = y_cur < y_limit[0] or y_cur > y_limit[1]
                is_z_out = z_cur < z_limit[0] or z_cur > z_limit[1]
                is_robot_out = is_x_out or is_y_out or is_z_out
                ## we don't want it our immediately ## ??
                if is_robot_out:
                    print('robot out:', i_step)
                    break


        print('ee:', ee_poses_ep[-1])
        print('noise:', noise)

    assert len(acts_ep) == len(ee_poses_ep) -1 == len(imgs_ep) - 1 # == rand_ep_len
    assert type(acts_ep[0]) == type(ee_poses_ep[0]) == type(imgs_ep[0]) == np.ndarray

    # mpy.show_images(imgs_ep, columns=8)
    return imgs_ep, acts_ep, ee_poses_ep