import os
import sys
from train import vs_code_debug

# from train_experiments.sh
exp_name = "20200122_debug_aggrevate"
job_name = "overfit_batch_100_debug_efficient_rotation"
config_file = "/home/hoyeung/Documents/vnla/code/tasks/VNLA/configs/experiment.json"

# from scripts/define_vars.sh
# always local
PT_DATA_DIR = "/home/hoyeung/blob_matterport3d/"
PT_OUTPUT_DIR_ALL="/home/hoyeung/blob_experiments/asknav/output_local/"

# -----------------------------------------------------------------------
# set env variables
os.environ["PT_DATA_DIR"] = PT_DATA_DIR
os.environ["PT_OUTPUT_DIR_ALL"] = PT_OUTPUT_DIR_ALL
os.environ["PT_OUTPUT_DIR"] = "{}/{}/{}".format(PT_OUTPUT_DIR_ALL, exp_name, job_name)

print (os.system("pwd"))
print ("making new output directory {}".format(os.environ['PT_OUTPUT_DIR']))
os.system("mkdir -p $PT_OUTPUT_DIR")

args = {}
args['config_file'] = config_file
args['exp_name'] = exp_name
args['job_name'] = job_name

# extras here!
# args['start_beta'] = 1.0
# args['n_iters'] = 1000
# data must be larger than batch_size
args['batch_size'] = 100
args['start_beta_decay'] = 100
args['decay_beta_every'] = 100
args['min_history_to_learn'] = 1000

args['save_every'] = 50
args['log_every'] = 50
args['plot_to_philly'] = 0

args['dropout_ratio'] = 0.0
args['agent_end_criteria'] = 2.5

# args['data_suffix'] = 'small_three_goals_short'
args['data_suffix'] = 'small_three_houses_single_goal_short'
# args['data_suffix'] = 'original'

vs_code_debug(args)