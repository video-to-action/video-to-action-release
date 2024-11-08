import sys
sys.path.insert(0, '.')
import os, torch, pdb, random
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import numpy as np
from os.path import join
import diffuser.datasets as datasets ## 
import diffuser.utils as utils
from datetime import datetime
import os.path as osp
from importlib import reload
from diffuser.libero.lb_eval_utils import load_lb_diffusion
from diffuser.libero.lb_eval_helper import LB_DP_Eval
from environment.libero.lb_env_v3 import LiberoEnvList_V3

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'
    vid_var_temp: float = 1.0
    dp_n_ddim_steps: int = 8
    eval_seed: int = None
    vis_gif: int = 1 # 1:vis all, 0: vis none


def main(args_train, args):
    if args.eval_seed is not None and args.eval_seed != -1:
        torch.backends.cudnn.deterministic = True
        random.seed(args.eval_seed)
        np.random.seed(args.eval_seed)
        torch.manual_seed(args.eval_seed)
    else:
        args.eval_seed = None
    
    utils.print_color(f'eval_seed: {args.eval_seed}', c='y')

    torch.backends.cudnn.allow_tf32 = getattr(args_train, 'allow_tf32', True)
    torch.backends.cuda.matmul.allow_tf32 = getattr(args_train, 'allow_tf32', True)

    ## savepath of the eval result
    args.savepath = osp.join(args.savepath, 
                             f"{datetime.now().strftime('%y%m%d-%H%M%S')}" +
                             f"-nm{args.plan_n_maze}-evSd{args.eval_seed}")
    is_sbatch = False ## can set to True for parallel slurm launch
    if is_sbatch: ## prevent duplicate folder name in sbatch parallel launch
        args.savepath = get_savepath_millisec(args)
    os.makedirs(args.savepath)
    
    #---------------------------------- loading ----------------------------------#
    
    ld_config = dict(trainer_fp16='fp16', trainer_amp=True) 
    ds, gcp_model, video_model, ema, trainer, epoch = load_lb_diffusion(args.logbase, args.dataset, args_train.exp_name, args=args_train, epoch=args.diffusion_epoch, ld_config=ld_config)
    accelerator = trainer.accelerator
    device = accelerator.device

    utils.print_color(f'{epoch=}', c='y')


    ## --------------------------------------
    ## ------- Eval Custom Settings ---------

    from diffuser.models.video_model import Video_PredModel
    video_model: Video_PredModel
    env_list: LiberoEnvList_V3 = ds.env_list

    ## setup hyperparam for the ema model (we use ema for inference)
    video_model.ema.ema_model.var_temp = args.vid_var_temp
    assert video_model.ema.ema_model.var_temp == 1.0
    ema.ema_model.num_inference_steps_ddim = args.dp_n_ddim_steps
    ema.ema_model.ddpm_var_temp = 0.5

    trainer.n_acts_per_pred = 8
    ema.ema_model.n_action_steps = 8

    ## --------------------------------------

    eval_cls = LB_DP_Eval

    eval_helper = eval_cls(
        gcp_model=None, # gcp_model, we do not use that one
        ema=ema,
        video_model=video_model,
        trainer=trainer,

        env_list=env_list,
        task_list=ds.task_list,
        cam_list=ds.cam_list,
        
        valid_seeds=list(range(100,100+int(args.plan_n_maze))), # 25
        ## e.g., valid_seeds=[0,1,2], seed for reset the env
        max_episode_steps=500, ## Dummy
        render_img_size=trainer.render_img_size,
        rendered_imgs_preproc_fn=trainer.rendered_imgs_preproc_fn,
        is_video_ddim=args.is_video_ddim,
        is_dp_ddim=args.is_dp_ddim,
        eval_n_preds_betw_vframes=args.eval_n_preds_betw_vframes,

        save_path=args.savepath,
        vid_use_autocast=True,
        ## NEW
        num_vid_pred_per_ep=args.num_vid_pred_per_ep,
        use_vid_first_n_frames=args.use_vid_first_n_frames,

        device=device,
    )

    eval_results = eval_helper.run_evals(vis_gif=args.vis_gif,)
    print(eval_results)
    suc_rate = eval_results['suc_rate']
    num_evals = eval_results['num_evals']
    epoch = int(epoch)
    eval_results['epoch'] = epoch
    eval_results['vid_var_temp'] = args.vid_var_temp
    eval_results['dp_var_temp'] = gcp_model.ddpm_var_temp
    vid_ds = video_model.ema.ema_model.sampling_timesteps if args.is_video_ddim else 100
    dp_ds = ema.ema_model.num_inference_steps_ddim if args.is_dp_ddim else 100
    eval_results['vid_diffusion'] = args_train.vid_diffusion['ckpts_dir']
    eval_results['eval_n_preds_betw_vframes'] = args.eval_n_preds_betw_vframes
    eval_results['eval_seed'] = args.eval_seed

    epoch_str = f'{round(epoch/1000)}k'
    
    json_path = osp.join(args.savepath, f"result-nm{num_evals}-sr{suc_rate*100:.1f}"
                         f"-ds{dp_ds}-vidDs{vid_ds}-ep{epoch_str}"
                         f"-vpep{args.num_vid_pred_per_ep}-vfn{args.use_vid_first_n_frames}"
                         f"-evSd{args.eval_seed}"
                         f".json")


    utils.save_json(eval_results, json_path)


def get_savepath_millisec(args):
    tmp_path = '/'.join(args.savepath.split('/')[:-1])
    return osp.join(tmp_path, f"{datetime.now().strftime('%y%m%d-%H%M%S-%f')}" +
                             f"-nm{args.plan_n_maze}-evSd{args.eval_seed}")


if __name__ == '__main__':
    args_train = Parser().parse_args('diffusion')
    args = Parser().parse_args('plan')
    
    args.is_video_ddim = False # if video model use ddim
    args.is_dp_ddim = True # if diffusion policy use ddim
    ## how many policy prediction for each video frame
    args.eval_n_preds_betw_vframes = 5

    ## number of video prediction for each problem
    args.num_vid_pred_per_ep = 5 # 1
    args.use_vid_first_n_frames = 2
    
    ## args.eval_seed = 0 ## manual seed setup
    ## if do visualization
    args.vis_gif = bool(args.vis_gif)

    main(args_train, args)
