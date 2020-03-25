import os
import sys
from pretrain_policy import vs_code_debug

# from train_experiments.sh
exp_name = "20200325_debug_pretraining"
job_name = "swap_classifier_image_frames_only"
config_file = "/home/hoyeung/Documents/vnla/code/tasks/VNLA/configs/pretrain_data.json"

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
args['n_iters'] = 200000

args['batch_size'] = 100

# args['lr'] = 1e-3

args['save_every'] = 1000 # 50
args['log_every'] = 1000  # 50
args['plot_to_philly'] = 0

# args['dropout_ratio'] = 0.5

args['seed'] = 42

args['include_language'] = False
args['include_actions'] = False

vs_code_debug(args)