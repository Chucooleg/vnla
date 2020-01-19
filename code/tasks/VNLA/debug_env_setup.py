import os
import sys
from train import vs_code_debug

# from train_experiments.sh
exp_name = "20200119_debug_aggrevate"
job_name = "train_batch_100_data_101"
config_file = "/home/hoyeung/Documents/vnla/code/tasks/VNLA/configs/experiment.json"

# from scripts/define_vars.sh
# always local
PT_DATA_DIR = "/home/hoyeung/blob_matterport3d/"
PT_OUTPUT_DIR = "/home/hoyeung/blob_experiments/asknav/output_local/"

# -----------------------------------------------------------------------
# set env variables
os.environ["PT_DATA_DIR"] = PT_DATA_DIR
os.environ["PT_OUTPUT_DIR"] = PT_OUTPUT_DIR
os.environ["OUTPUT_DIR"] = "{}/{}/{}".format(PT_OUTPUT_DIR, exp_name, job_name)

print (os.system("pwd"))
print ("making new output directory {}".format(os.environ['OUTPUT_DIR']))
os.system("mkdir -p $OUTPUT_DIR")

args = {}
args['config_file'] = config_file
args['exp_name'] = exp_name
args['job_name'] = job_name

# extras here!
# args['start_beta'] = 1.0
# args['n_iters'] = 1000
# data must be larger than batch_size
args['batch_size'] = 100
args['start_beta_decay'] = 5
args['decay_beta_every'] = 5
args['min_history_to_learn'] = 1000
args['max_buffer_size'] = 10000

args['save_every'] = 5
args['log_every'] = 5
args['plot_to_philly'] = 0

args['data_suffix'] = 'small_three_houses_single_goal_short'

vs_code_debug(args)
