# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

def make_parser():
    parser = argparse.ArgumentParser()

    # main python call
    parser.add_argument('-config_file', type=str, 
        help='configuration file')
    parser.add_argument('-load_path', type=str,
        help='path to load a pretrained model')
    parser.add_argument('-exp_name', type=str,
        help='name of the experiment')   
    parser.add_argument('-job_name', type=str,
        help='name of the job')  

    # take care of any extras from main python call here

    # Baselines
    parser.add_argument('-uncertainty_handling', type=str,
        help='options to handle agent uncertainty')

    # Vocab
    parser.add_argument('-external_main_vocab', type=str,
        help='provide a different vocab file')

    # Advisor
    parser.add_argument('-ask_advisor', type=str,
        help="type of ask advisor ('direct' or 'verbal_easy' or 'verbal_hard'")
    parser.add_argument('-subgoal_vocab', type=str,
        help='subgoal vocabulary')

    # Bootstrapping settings
    parser.add_argument('-bootstrap', type=int, default=1,
        help='bootstrap on (1) or off (0)')
    parser.add_argument('-n_ensemble', type=int, default=10,
        help='number of bootstrap heads')
    parser.add_argument('-bernoulli_probability', type=float, default=1.0,
        help='bernoulli probability that a datapt is exposed to backprop a head')
    parser.add_argument('-bootstrap_majority_vote', type=int, default=1,
        help='majority vote among heads (1) or sampling heads (0)')

    return parser