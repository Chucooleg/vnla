#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

source define_vars.sh

cd ../

exp_name=$1
# extras=${2:-0}
extras=${@:2}

config_file="configs/verbal_hard.json"
output_dir="main_$exp_name"

ask=""

if [ "$exp_name" == "none" ]
then
  ask="-no_ask 1"
elif [ "$exp_name" == "first" ]
then
  ask="-ask_first 1"
elif [ "$exp_name" == "random" ]
then
  ask="-random_ask 1"
elif [ "$exp_name" == "teacher" ]
then
  ask="-teacher_ask 1"
elif [ "$exp_name" == "learned" ]
then
  ask=""
else
  # echo "Usage: bash train_main_results.sh [none|first|random|teacher|learned] [gpu_id]"
  # echo "Example: bash train_main_results.sh learned 0"
  echo "Usage: bash train_main_results.sh [none|first|random|teacher|learned] [extra arguments]"
  echo "Example: bash train_main_results.sh learned -bootstrap 1 -n_ensemble 10 -bernoulli_probability 0.5 -bootstrap_majority_vote 1 -gradient_clipping 0"
  exit
fi

# command="python -u -m pdb -c continue train.py -config $config_file -exp $output_dir $ask $extras"
# command="python -u train.py -config $config_file -exp $output_dir $ask $extras"
command="python -u -m torch.utils.bottleneck train.py -config $config_file -exp $output_dir $ask $extras"

echo $command
$command






