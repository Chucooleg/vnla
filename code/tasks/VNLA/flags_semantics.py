# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-local_run', type=int, 
        help='local run on/off, 1 or 0.')

    # override main python call in train_*.sh
    parser.add_argument('-config_file', type=str, 
        help='configuration file')
    parser.add_argument('-load_path', type=str,
        help='path to load a pretrained model')
    parser.add_argument('-exp_name', type=str,
        help='name of the experiment')   
    parser.add_argument('-job_name', type=str,
        help='name of the job')

    # Dataset
    parser.add_argument('-seed', type=int,
        help='random seed')
    parser.add_argument('-img_features', type=str,
        help='path to pretrained image embeddings')
    parser.add_argument('-img_feature_size', type=int,
        help='image embedding size')
    parser.add_argument('-img_extent', type=str,
        help='how much each prediction covers the panoramic sphere. "single" for single frame, "vertical" for full vertical 3 frames, "full" for all 36 frames in the sphere.')        
    parser.add_argument('-n_unseen_scans', type=int,
        help='number of scans to reserve for unseen validation')
        
    parser.add_argument('-scans_path', type=str,
        help='path to text file that stores all original asknav training scans')   
    parser.add_argument('-room_types_path', type=str,
        help='path to text file that stores room types in original asknav training scans')

    parser.add_argument('-tr_idx_save_path', type=str,
        help='path to text file that stores training long_id, viewix and label')   
    parser.add_argument('-val_seen_idx_save_path', type=str,
        help='path to text file that stores val seen long_id, viewix and label')
    parser.add_argument('-val_unseen_idx_save_path', type=str,
        help='path to text file that stores val unseen long_id, viewix and label')

    # Training Loop parameters
    parser.add_argument('-n_epochs', type=int,
        help='number of training epochs.')
    parser.add_argument('-batch_size', type=int,
        help='rollout batch size, used to compute val loss as well.') 
    parser.add_argument('-save_every', type=int,
        help='number of epoch between model savings')

    # Optimizer parameters
    parser.add_argument('-lr', type=float,
        help='learning rate')
    parser.add_argument('-weight_decay', type=float,
        help='L2-regularization weight')
    parser.add_argument('-start_lr_decay', type=int,
        help='iteration to start decaying learning rate')
    parser.add_argument('-lr_decay_rate', type=float,
        help='learning rate decay rate')
    parser.add_argument('-decay_lr_every', type=int,
        help='number of iterations between learning rate decays')

    # classifier parameters
    parser.add_argument('-layers', type=int,
        help='number of layers for classifier, min 1.')
    parser.add_argument('-dropout_ratio', type=float,
        help='dropout ratio float.')

    # tensorboard logging
    parser.add_argument('-plot_to_philly', type=int,
        help='plot to philly web interface on (1) or off (0)')

    # Evaluation
    parser.add_argument('-eval_only', type=int,
        help='evaluation mode')

    return parser