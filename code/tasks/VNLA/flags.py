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
    parser.add_argument('-data_suffix', type=str,
        help='name of the custom data for training e.g. ask_nav_train_[data_suffix].json')  
    parser.add_argument('-eval_data_suffix', type=str,
        help='name of the custom data for validation/eval e.g. ask_nav_val_[data_suffix].json')  

    # Vocab (not in master config)
    parser.add_argument('-external_main_vocab', type=str,
        help='provide a different vocab file path')

    # Dataset
    parser.add_argument('-seed', type=int,
        help='random seed')
    parser.add_argument('-data_dir', type=str,
        help='data directory')
    parser.add_argument('-img_features', type=str,
        help='path to pretrained image embeddings')
    parser.add_argument('-img_feature_size', type=int, default=2048,
        help='image embedding size')

    # Choice of algorithm
    parser.add_argument('-interaction_mode', type=str,
        help='expert_agent, human_agent, or none_agent')
    parser.add_argument('-uncertainty_handling', type=str,
        help='whether agent can ask questions or not. no_ask or learned_ask_flag')  
    parser.add_argument('-recovery_strategy', type=str,
        help='whether agent can recover or not. no_recovery or learned_recovery') 
    parser.add_argument('-navigation_objective', type=str,
        help='navigation objective. either "value_estimateion" or "action_imitation"') 
    parser.add_argument('-ask_advisor', type=str,
        help="type of ask advisor ('direct' or 'verbal'")
    parser.add_argument('-nav_feedback', type=str,
        help='navigation training method (deprecated)')
    parser.add_argument('-ask_feedback', type=str,
        help='asking training method (deprecated)')
    parser.add_argument('-recover_feedback', type=str,
        help='recover training method (deprecated)')

    # Meta-level algorithm parameters
    parser.add_argument('-max_episode_length', type=int,
        help='maximum number of actions per epsiode')
    parser.add_argument('--deviate_threshold', type=float)
    parser.add_argument('--uncertain_threshold', type=float)
    parser.add_argument('--unmoved_threshold', type=int)      
    parser.add_argument('-success_radius', type=float,
        help='success radius')
    parser.add_argument('-agent_end_criteria', type=float,
        help='threshold of q value estimate for agent to end episode')

    # Learned asking parameters
    parser.add_argument('-query_ratio', type=float,
        help='ratio between number of steps the agent is assisted and total number of steps (tau)')   
    parser.add_argument('-max_ask_budget', type=int,
        help='budget upperbound')
    parser.add_argument('-budget_embed_size', type=int,
        help='ask budget embed dim integer.')
    parser.add_argument('-n_subgoal_steps', type=int,
        help='number of next actions suggested by a subgoal')
    # (Not in master config)
    parser.add_argument('-rule_a_e', type=int,
        help='Use rules (a) and (e) only for help-requesting teacher')
    parser.add_argument('-rule_b_d', type=int,
        help='Use rules (b) to (d) only for help-requesting teacher')
    # Don't touch these ones
    parser.add_argument('-backprop_softmax', type=int, default=1)
    parser.add_argument('-backprop_ask_features', type=int)

    # Aggrevate parameters
    parser.add_argument('-start_beta', type=float,
        help='aggrevate: expert rollin probability beta to start with, between 0.0 and 1.0')
    parser.add_argument('-start_beta_decay', type=int,
        help='aggrevate: minimum number of iterations before beta decay')
    parser.add_argument('-beta_decay_rate', type=float,
        help='aggrevate: exponential decay rate for beta, between 0.0 and 1.0')
    parser.add_argument('-decay_beta_every', type=int,
        help='aggrevate: number of iterations between calling decay on beta')
    parser.add_argument('-min_history_to_learn', type=int,
        help='aggrevate: minimum history buffer size before training can start')
    parser.add_argument('-num_recent_frames', type=int,
        help='aggrevate: number of steps in recent history for LSTM unfold through, before predicting q-values for current time steps')      

    # Sort by groundtruth parameters
    parser.add_argument('-sort_by_ground_truth', type=str,
        help='1/0. Whether to sort training data by ground truth distance-to-go') 
    parser.add_argument('-start_samp_bias', type=float,
        help='sampling bias to start with, between 0.0 and 1.0')
    parser.add_argument('-start_samp_bias_decay', type=int,
        help='minimum number of iterations before sampling bias decay')
    parser.add_argument('-samp_bias_decay_rate', type=float,
        help='exponential decay rate for sampling bias, between 0.0 and 1.0')
    parser.add_argument('-decay_samp_bias_every', type=int,
        help='number of iterations between calling decay on sampling bias')

    # History buffer parameters
    parser.add_argument('-max_buffer_size', type=int,
        help='maximum number of examples history buffer can save')   

    # Training Loop parameters
    parser.add_argument('-n_iters', type=int,
        help='number of training iterations (batches)')
    parser.add_argument('-batch_size', type=int,
        help='rollout batch size, used to compute val loss as well.')
    parser.add_argument('-train_batch_size', type=int,
        help='training batch size, iid examples sampled from history buffer.')    
    parser.add_argument('-save_every', type=int,
        help='number of iterations between model savings')
    parser.add_argument('-log_every', type=int,
        help='number of iterations between information loggings') 

    # Optimizer parameters
    parser.add_argument('-loss_function', type=str,
        help='loss function. `l1` or `l2`')
    parser.add_argument('-norm_loss_by_dist', type=int,
        help='1/0. Whether to normalize loss by gold distance-to-goal.')
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

    # Vocabulary parameters
    parser.add_argument('-subgoal_vocab', type=str,
        help='subgoal vocabulary')
    parser.add_argument('-min_word_count', type=str,
        help='minimum word count cutoff when building vocabulary')
    parser.add_argument('-split_by_spaces', type=int,
        help='split word by spaces (always set true)')          

    # LSTM parameters
    parser.add_argument('-word_embed_size', type=int,
        help='word embed dim integer.')
    parser.add_argument('-nav_embed_size', type=int,
        help='nav embed dim integer.')      
    parser.add_argument('-ask_embed_size', type=int,
        help='ask embed dim integer.') 
    parser.add_argument('-max_input_length', type=int,
        help='max input length for language input tokens.') 
    parser.add_argument('-num_lstm_layers', type=int,
        help='number of lstm layers.') 
    parser.add_argument('-hidden_size', type=int,
        help='hidden size for LSTM cell.') 
    parser.add_argument('-coverage_size', type=int,
        help='coverage size for attention mechanism.') 
    parser.add_argument('-bidirectional', type=int,
        help='whether to use (1/0) bidirectional LSTM.') 
    parser.add_argument('-dropout_ratio', type=float,
        help='dropout ratio float.') 

    # bootstrap parameters
    parser.add_argument('-bootstrap', type=int,
        help='bootstrap on (1) or off (0)')
    parser.add_argument('-n_ensemble', type=int,
        help='number of bootstrap heads')
    parser.add_argument('-bernoulli_probability', type=float,
        help='bernoulli probability that a datapt is exposed to backprop a head')

    # tensorboard logging
    parser.add_argument('-plot_to_philly', type=int,
        help='plot to philly web interface on (1) or off (0)')

    # Evaluation (Not in master config)
    parser.add_argument('-eval_only', type=int,
        help='evaluation mode')
    parser.add_argument('-multi_seed_eval', type=int,
        help='evaluate with multiple seeds (automatically set -eval_only 1)')

    # Others (Not in master config)
    parser.add_argument('-device_id', type=int, default=0,
        help='gpu id')
    

    return parser