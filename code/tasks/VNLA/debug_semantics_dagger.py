import os
import sys
from train import vs_code_debug

# from train_experiments.sh
exp_name = "20200305_debug_dagger_no_language"
job_name = "debug_1"
config_file = "/home/hoyeung/Documents/vnla/code/tasks/VNLA/configs/verbal_hard.json"

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
args['n_iters'] = 5000

args['batch_size'] = 100

# args['lr'] = 1e-3

args['save_every'] = 100 # 50
args['log_every'] = 100  # 50
args['plot_to_philly'] = 0

# args['dropout_ratio'] = 0.5

args['seed'] = 42

args['no_ask'] = 1

# args['data_suffix'] = "small_three_houses_single_goal_short"
args['data_suffix'] = "original"


vs_code_debug(args)