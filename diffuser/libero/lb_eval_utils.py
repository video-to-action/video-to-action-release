import os, pdb
from tqdm import tqdm
import numpy as np
####
from diffuser.libero.lb_train_utils import LB_Init_Trainer
from diffuser.diffusion_policy import Init_Diffusion_Policy
from diffuser.utils.serialization import DiffusionExperiment_Feb19, load_config, get_latest_epoch


def load_lb_diffusion(*loadpath, args, epoch='latest', device='cuda:0', ld_config={}):
    ''' 
    Load out the models for eval
    '''
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    trainer_config._dict['fp16'] = ld_config['trainer_fp16']
    trainer_config._dict['amp'] = ld_config['trainer_amp']
    ## enable loading a placeholder full random action dataset if not downloaded
    trainer_config._dict['trainer_dict']['cur_mode'] = 'eval'

    dataset = dataset_config()
    
    from diffuser.libero.lb_video_model_utils import lb_get_video_model_gcp_v2
    
    video_model = lb_get_video_model_gcp_v2(**args.vid_diffusion)

    init_tr = LB_Init_Trainer(args)
    if type(getattr(args, 'resnet_cfg', None)) == dict:
        raise NotImplementedError
    else:
        init_dp = Init_Diffusion_Policy(args)

    trainer = trainer_config(
        init_diff_policy=init_dp,
        video_model=video_model,
        tokenizer=init_tr.tokenizer,
        text_encoder=init_tr.text_encoder,
        train_set=dataset,
        valid_set=dataset, # useless
    )

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    trainer.load(epoch)
    from diffuser.models.train_utils import freeze_model
    gcp_model = init_dp.diffusion_policy
    freeze_model(gcp_model)
    freeze_model(trainer.ema.ema_model)

    return DiffusionExperiment_Feb19(dataset, gcp_model, video_model, trainer.ema, trainer, epoch)