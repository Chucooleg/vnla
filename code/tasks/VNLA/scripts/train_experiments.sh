#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

source define_vars.sh

cd ../

exp_name=$1     # (date_?)
job_name=$2     # philly job name
config_file=$3  # "configs/experiment.json"
extras=${@:4} 

export PT_OUTPUT_DIR=$PT_OUTPUT_DIR_ALL/$exp_name/$job_name
echo making new output directory $PT_OUTPUT_DIR
mkdir -p $PT_OUTPUT_DIR

# mine
# command="python -u -m pdb -c continue train.py ..."
# command="python -u -m torch.utils.bottleneck train.py ..."
# command="python -u -m cProfile -o $PT_OUTPUT_DIR/restats train.py ..."
# command="python -u -m cProfile -o $PT_OUTPUT_DIR/restats train.py -exp $exp_name -job $job_name -config $config_file $extras"
command="python -u -m pdb -c continue train.py -exp $exp_name -job $job_name -config $config_file $extras"

echo $command
$command