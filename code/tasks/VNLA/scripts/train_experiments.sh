#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# export PT_DATA_DIR=/home/hoyeung/blob_matterport3d/
# export PT_OUTPUT_DIR=/home/hoyeung/blob_experiments/asknav/output_local/
# export PT_DATA_DIR=/mnt/matterport3d/
# export PT_OUTPUT_DIR=/mnt/experiment-results-philly/asknav/output_philly/
source define_vars.sh

cd ../

exp_name=$1     # (date_?)
job_name=$2     # philly job name
config_file=$3  # "configs/experiment.json"
extras=${@:4} 

export OUTPUT_DIR=$PT_OUTPUT_DIR/$exp_name/$job_name
echo making new output directory $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

# mine
# command="python -u -m pdb -c continue train.py ..."
# command="python -u -m torch.utils.bottleneck train.py ..."
# command="python -u -m cProfile -o $OUTPUT_DIR/restats train.py ..."
command="python -u -m cProfile -o $OUTPUT_DIR/restats train.py -exp $exp_name -job $job_name -config $config_file $extras"

echo $command
$command