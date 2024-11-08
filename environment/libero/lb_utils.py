import os
from libero.libero import benchmark
from libero.libero import benchmark, get_libero_path, set_libero_default_path
import diffuser.utils as utils

def lb_full_cam_name(cam_name, is_depth=False):
    '''translate short cam name to name in the obs'''
    assert cam_name in ['agent', 'gripper']
    if is_depth:
        if cam_name == 'agent':
            view_name = 'agentview_depth'
        elif cam_name == 'gripper':
            view_name = 'robot0_eye_in_hand_depth'
    else:
        if cam_name == 'agent':
            view_name = 'agentview_image'
        elif cam_name == 'gripper':
            view_name = 'robot0_eye_in_hand_image'
    
    return view_name


def lb_task_idx_to_infos(task_suite_name, task_idx_list):
    ## keys: 'libero_spatial', 'libero_object', 'libero_90', ...
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    
    task_list = [] # list of str for model input
    task_dirname_list = []
    task_bddl_file_list = [] # a list of str, filepath
    
    for task_id in task_idx_list:
        # retrieve a specific task
        task = task_suite.get_task(task_id)
        task_name = task.name # 'LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate'
        task_dirname_list.append(task_name)
        
        ## input to the model
        task_description = task.language
        task_list.append(task_description) # 'put the red mug on the left plate'
        ## bddl file is used to init envs
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        task_bddl_file_list.append(task_bddl_file)
        utils.print_color(f"[info] retrieving task {task_id} from suite {task_suite_name}," 
                            f"the language instruction is {task_description}")
    
    return task_list, task_dirname_list, task_bddl_file_list


def lb_extract_imgs_from_batch_obs(batch_obs, cam_name, is_depth):
    '''return a list of numpy imgs (128,128,3)'''
    cam_name = lb_full_cam_name(cam_name, is_depth=is_depth)
    imgs = []
    for obs in batch_obs:
        imgs.append(obs[cam_name])
    return imgs

def lb_merged_str_idx_tk_list(task_idx_list, task_list):
    '''
    return like 65-put-xxxxx
    '''
    assert len(task_idx_list) == len(task_list)
    mg_list = []
    for i_t, tk in enumerate(task_list):
        tk = tk.replace(' ', '-')
        tmp_str = f"{task_idx_list[i_t]}-{tk}"
        mg_list.append(tmp_str)

    return mg_list


def lb_merged_str_idx_tk_1(env_list, task_name):
    '''
    return like 65-put-xxxxx
    '''
    task_idx = env_list.task_to_task_idx[task_name]
    task_name = task_name.replace(' ', '-')
    tmp_str = f"{task_idx}-{task_name}"

    return tmp_str


# def lb_merged_str_idx_tk_2(task_idx, task_name):
#     '''
#     return like 65-put-xxxxx
#     '''
#     task_name = task_name.replace(' ', '-')
#     tmp_str = f"{task_idx}-{task_name}"

#     return tmp_str