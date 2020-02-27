#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

source define_vars.sh

cd ../

exp_name=$1     # (date_?)
job_name=$2     # philly job name
config_file=$3  # "configs/experiment.json"
load_path=$4
eval_data_suffix=$5

export PT_EXP_DIR=$PT_OUTPUT_DIR/output_philly/$exp_name/$job_name
echo making new output directory $PT_EXP_DIR
mkdir -p $PT_EXP_DIR

export PT_TENSORBOARD_DIR=$PT_OUTPUT_DIR/vis_files/$exp_name/$job_name
echo making new tensorboard directory $PT_TENSORBOARD_DIR
mkdir -p $PT_TENSORBOARD_DIR

# mine
# command="python -u -m torch.utils.bottleneck train.py ..."
# command="python -u -m cProfile -o $PT_EXP_DIR/restats train.py ..."
# command="python -u -m cProfile -o $PT_EXP_DIR/restats train.py -exp $exp_name -job $job_name -config $config_file $extras"
command="python -u train.py -exp $exp_name -job $job_name -config $config_file -multi_seed_eval 1 -load_path $load_path -eval_data_suffix $eval_data_suffix"

echo $command
$command