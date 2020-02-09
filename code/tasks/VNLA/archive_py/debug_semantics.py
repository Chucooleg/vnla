import os
import sys
from train_semantics import vs_code_debug

# from train_experiments.sh
exp_name = "20200203_debug_semantics"
job_name = "debug_local_vertical"
config_file = "/home/hoyeung/Documents/vnla/code/tasks/VNLA/configs/experiments_semantic_classifier.json"

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
args['n_unseen_scans'] = 1
args['n_epochs'] = 100
# data must be larger than batch_size
args['batch_size'] = 100
args['lr'] = 1e-4

args['save_every'] = 10 # 50s
# args['log_every'] = 2  # 50
args['plot_to_philly'] = 0

args['dropout_ratio'] = 0.5
args['seed'] = 42
args['layers'] = 2

# single, vertical or full
args['image_extent'] = 'vertical'
args['tr_idx_save_path'] = 'semantics/asknav_train_full_vertical_indices.txt'
args['val_seen_idx_save_path'] = 'semantics/asknav_val_seen_full_vertical_indices.txt'
args['val_unseen_idx_save_path'] = 'semantics/asknav_val_unseen_full_vertical_indices.txt'

vs_code_debug(args)



#"""  - cd tasks/VNLA/scripts/ && bash train_semantic_classifier.sh "20200203_philly_debug_semantics_next_vertex" "next_vertex_classifier_seed_42_batch_size_100_lr_0.0001_layers_1" "configs/experiments_semantic_classifier.json" -batch_size 100 -layers 1 -save_every 10 -plot_to_philly 0 -dropout_ratio 0.5 -lr 0.0001 -n_epochs 200 -seed 42 -tr_idx_save_path "semantics/asknav_train_next_vertex_indices.txt" -val_seen_idx_save_path "semantics/asknav_val_seen_next_vertex_indices.txt" -val_unseen_idx_save_path "semantics/asknav_val_unseen_next_vertex_indices.txt""""