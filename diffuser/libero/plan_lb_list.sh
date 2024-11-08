#!/bin/bash
source ~/.bashrc
source activate v2a_libero_release

## a list of config files to be evaluated
config_list=( \
    "config/libero/lb_tk8_65to72.py" \
)

epoch='latest' ## use the latest checkpoint

{
## some other hyper-parameters options are inside plan_lb.py
for config in "${config_list[@]}"; do
    CUDA_VISIBLE_DEVICES=${2:-0} OMP_NUM_THREADS=1 \
    python3 diffuser/libero/plan_lb.py --config $config --plan_n_maze ${1:-25} \
    --diffusion_epoch $epoch \
    --vid_var_temp 1.0 \
    --eval_seed 0 \
    --vis_gif 1 \

done

echo "Done"
exit 0
}
