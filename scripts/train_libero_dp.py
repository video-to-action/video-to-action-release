import sys
sys.path.append('.'); sys.path.insert(0, './')
import torch, wandb, pdb
import diffuser.utils as utils
import os.path as osp
from diffuser.libero.lb_video_model_utils import lb_get_video_model_gcp_v2
from diffuser.libero.lb_train_utils import LB_Init_Trainer
from diffuser.diffusion_policy import Init_Diffusion_Policy

torch.backends.cudnn.benchmark = True

assert __name__ == '__main__'

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = ''
    config: str = ''

## dataset is set inside parse_args automatically
args = Parser().parse_args('diffusion')

torch.backends.cudnn.allow_tf32 = getattr(args, 'allow_tf32', True)
torch.backends.cuda.matmul.allow_tf32 = getattr(args, 'allow_tf32', True)

utils.print_color('args.dataset', args.dataset, 'luotest-color')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    ##
    env=args.dataset,
    target_size=args.input_img_size,
    dataset_config=args.dataset_config,
)

dataset = dataset_config()

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

## ------------------ Video Diffusion Model ------------------
assert args.input_img_size == (128, 128)
args.vid_diffusion['target_size'] = args.input_img_size
video_model = lb_get_video_model_gcp_v2(**args.vid_diffusion)

# pdb.set_trace()

## ------------------ Goal-Conditioned Policy ------------------

init_diff_policy = Init_Diffusion_Policy(args)




## ------------------ Trainer ------------------
trainer_dict = getattr(args, 'trainer_dict', {})
init_tr = LB_Init_Trainer(args)


trainer_config = utils.Config( # V7 trainer
    init_tr.trainer_cls,
    savepath=(args.savepath, 'trainer_config.pkl'),

    channels=3,
    train_batch_size=trainer_dict['batch_size'],
    video_batch_size=trainer_dict.get('batch_size_v', 4),
    valid_batch_size=1,

    gradient_accumulate_every=args.gradient_accumulate_every,
    augment_horizontal_flip=None,

    train_num_steps = args.n_train_steps,

    opt_params=args.opt_params,
    ema_params=args.ema_params,

    render_img_size=args.render_img_size, # (320, 240),
    input_img_size=args.input_img_size, # (128, 128),
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    log_freq=args.log_freq,

    ## seems useless:
    n_samples=args.n_samples,
    results_folder=args.savepath,
    
    trainer_dict=trainer_dict,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

# print('args.diffusion:', args.diffusion)

trainer = trainer_config(
    init_diff_policy=init_diff_policy,
    video_model=video_model,
    tokenizer=init_tr.tokenizer,
    text_encoder=init_tr.text_encoder,
    train_set=dataset,
    valid_set=dataset, # useless
)

if getattr(args, 'do_train_resume', False): # for a sample resume, should be good
    trainer.load(utils.get_latest_epoch( (trainer.results_folder,) ))

# pdb.set_trace()

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#
gcp_model = init_diff_policy.diffusion_policy
utils.report_parameters(gcp_model)


print('Testing forward...', end=' ', flush=True)
device = next(gcp_model.parameters()).device
img1,img2,_,act = dataset.sample_random_tensor(1,args.trainer_dict['model_act_horizon'],device)
batch = trainer.to_batch_dict(img1, img2, act)


loss = gcp_model.compute_loss(batch)
loss.backward()
trainer.opt.zero_grad()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- save config ---------------------------------#
#-----------------------------------------------------------------------------#


all_configs = dict(dataset_config=dataset_config._dict, 
                args=args.as_dict(),
                gcp_model_conf=init_diff_policy.all_conf,
                trainer_config=trainer_config._dict
                )
# print(args)
ckp_path = args.savepath
if trainer.accelerator.is_main_process:
    wandb.init(
        project="Video-to-Action-Release",
        name=args.logger_name,
        id=args.logger_id,
        dir=ckp_path,
        config=all_configs, ## need to be a dict
        # resume="must",
    )

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

trainer.train()