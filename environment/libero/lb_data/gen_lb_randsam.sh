#!/bin/bash
source ~/.bashrc
source activate v2a_libero_release


sub_conf='lb_randsam_8tk_perTk5'
# sub_conf='lb_randsam_8tk_perTk500'
sub_conf='lb_randsam_8tk_dummy_example'

{
MUJOCO_EGL_DEVICE_ID=${1:-0} \
MUJOCO_GL=egl \
CUDA_VISIBLE_DEVICES=${1:-0} \
python3 environment/libero/lb_data/lb_randsam.py \
    --sub_conf ${sub_conf}

exit 0
}

