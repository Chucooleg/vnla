import os
import sys
from train import vs_code_debug

# from train_experiments.sh
exp_name = "20200221_philly_aggrevate_costtogo_logging"
job_name = "trial_4"
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
args['multi_seed_eval'] = 1

args['config_file'] = config_file
args['exp_name'] = exp_name
args['job_name'] = job_name

# extras here!
args['load_path'] = '/home/hoyeung/blob_experiments/output_local/20200207_debug_aggrevate/debug_faster_tensor_loading_2/value_estimation_no_ask_no_recovery_last.ckpt'
args['multi_seed_eval'] = 1

# args['data_suffix'] = 'small_three_goals_short'
args['data_suffix'] = 'small_three_houses_single_goal_short'
# args['data_suffix'] = 'original'
# args['eval_data_suffix'] = 'small_ten_goals_short'


vs_code_debug(args)