#!/bin/bash

source ~/.bashrc
source activate v2a_libero_release

config="config/libero/lb_tk8_luotest.py" ## testing template
# config="config/libero/lb_tk8_65to72.py" ## pre-trained model provided


{
CUDA_VISIBLE_DEVICES=${1:-0} OMP_NUM_THREADS=1 \
python3 scripts/train_libero_dp.py --config $config

echo "Done"
exit 0

}
