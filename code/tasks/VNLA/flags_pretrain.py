# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

def make_parser():
   parser = argparse.ArgumentParser()

   parser.add_argument('-config_file', type=str,
        help='configuration file')
   parser.add_argument('-load_path', type=str,
        help='path to load a pretrained model')
   parser.add_argument('-exp_name', type=str,
        help='name of the experiment')
   parser.add_argument('-job_name', type=str,
        help='name of the job')
   parser.add_argument('-seed', type=int,
        help='random seed')
   parser.add_argument('-data_dir', type=str,
        help='data directory')
   parser.add_argument('-img_features', type=str,
        help='path to pretrained image embeddings')
   parser.add_argument('-img_feature_size', type=int, default=2048,
        help='image embedding size')
   parser.add_argument('-batch_size', type=int,
        help='batch size (both training and evaluation)')
   parser.add_argument('-dropout_ratio', type=float,
        help='dropout probability')
   parser.add_argument('-lr', type=float,
        help='learning rate')
   parser.add_argument('-weight_decay', type=float,
        help='L2-regularization weight')
   parser.add_argument('-n_iters', type=int,
        help='number of training iterations (batches)')
   parser.add_argument('-start_lr_decay', type=int,
        help='iteration to start decaying learning rate')
   parser.add_argument('-lr_decay_rate', type=float,
        help='learning rate decay rate')
   parser.add_argument('-decay_lr_every', type=int,
        help='number of iterations between learning rate decays')
   parser.add_argument('-save_every', type=int,
        help='number of iterations between model savings')
   parser.add_argument('-log_every', type=int,
        help='number of iterations between information loggings')

   # Evaluation
   parser.add_argument('-eval_only', type=int,
        help='evaluation mode')

   # tensorboard
   parser.add_argument('-plot_to_philly', type=int,
        help='plot to philly platform 1/0. default 0')

   return parser
