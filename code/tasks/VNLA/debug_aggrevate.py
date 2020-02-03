import os
import sys
from train import vs_code_debug

# from train_experiments.sh
exp_name = "20200128_debug_aggrevate"
job_name = "debug_pre_tunning_94798"
config_file = "/home/hoyeung/Documents/vnla/code/tasks/VNLA/configs/experiment.json"

# from scripts/define_vars.sh
# always local
PT_DATA_DIR = "/home/hoyeung/blob_matterport3d/"
PT_OUTPUT_DIR="/home/hoyeung/blob_experiments/output_local"

# -----------------------------------------------------------------------
# set env variables
os.environ["PT_DATA_DIR"] = PT_DATA_DIR
os.environ["PT_OUTPUT_DIR"] = PT_OUTPUT_DIR
os.environ["PT_EXP_DIR"] = "{}/{}/{}".format(PT_OUTPUT_DIR, exp_name, job_name)

print (os.system("pwd"))
print ("making new experiment directory {}".format(os.environ['PT_EXP_DIR']))
os.system("mkdir -p $PT_EXP_DIR")

args = {}
args['local_run'] = 1

args['config_file'] = config_file
args['exp_name'] = exp_name
args['job_name'] = job_name

# extras here!
# args['n_iters'] = 1000
# data must be larger than batch_size
args['batch_size'] = 100
args['train_batch_size'] = 200
args['start_beta'] = 1.0 # 1.0
args['beta_decay_rate'] = 0.98
args['start_beta_decay'] = 5
args['decay_beta_every'] = 5
args['min_history_to_learn'] = 1000
args['lr'] = 1e-3
args['loss_function'] = 'l1'

args['save_every'] = 10 # 50
args['log_every'] = 10  # 50
args['plot_to_philly'] = 0

args['dropout_ratio'] = 0.5
args['agent_end_criteria'] = 2.5
args['seed'] = 42

# args['data_suffix'] = 'small_three_goals_short'
# args['data_suffix'] = 'small_three_houses_single_goal_short'
args['data_suffix'] = 'original'

vs_code_debug(args)