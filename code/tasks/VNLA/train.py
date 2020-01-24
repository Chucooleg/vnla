from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import os
import time
import numpy as np
import pandas as pd
import argparse
import json
from collections import defaultdict, Counter
from argparse import Namespace
from pprint import pprint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy import stats

from utils import read_vocab, write_vocab, build_vocab, Tokenizer, padding_idx, timeSince
from env import VNLABatch, VNLAExplorationBatch
from model import AttentionSeq2SeqFramesModel, AttentionSeq2SeqContinuousModel
from action_imitation_no_ask_agent import ActionImitationNoAskAgent
from action_imitation_verbal_ask_agent import ActionImitationVerbalAskAgent
from value_estimation_no_ask_no_recovery_agent import ValueEstimationNoAskNoRecoveryAgent
from tensorboardX import SummaryWriter

from eval import Evaluation
from oracle import *
from flags import make_parser


def set_path():
    '''set paths for data, saving, logging'''

    # Set general output dir
    hparams.exp_dir = os.getenv('PT_OUTPUT_DIR')
    
    # Set model prefix
    hparams.bootstrap = 1 if hasattr(hparams, 'bootstrap') and hparams.bootstrap else 0
    hparams.model_prefix = "{}_{}_{}".format(hparams.navigation_objective, hparams.uncertainty_handling, hparams.recovery_strategy)
    if hparams.bootstrap:
        hparams.model_prefix = '{}_bootstrap'.format(hparams.model_prefix)

    # Set tensorboard log dir
    if hparams.plot_to_philly:
        hparams.tensorboard_dir = os.environ.get('PHILLY_LOG_DIRECTORY', '.')
    else:
        hparams.tensorboard_dir = os.environ.get('PT_OUTPUT_DIR', '.')
        # hparams.tensorboard_dir = os.path.join(hparams.exp_dir, "tensorboard")
        # if not os.path.exists(hparams.tensorboard_dir):
        #     os.makedirs(hparams.tensorboard_dir)
        #     os.makedirs(os.path.join(hparams.tensorboard_dir, 'main'))
        #     os.makedirs(os.path.join(hparams.tensorboard_dir, 'agent'))
    print ("tensorboard dir from hparams = {}".format(hparams.tensorboard_dir))

    # Set model load path
    hparams.load_path = hparams.load_path if hasattr(hparams, 'load_path') and \
        hparams.load_path is not None else \
        os.path.join(hparams.exp_dir, '%s_last.ckpt' % hparams.model_prefix)

    # Set history buffer load path
    hparams.history_buffer_path = hparams.history_buffer_path if (hasattr(hparams, 'history_buffer_path') \
        and hparams.history_buffer_path is not None) else os.path.join(hparams.exp_dir, 'history_buffer')
    if not os.path.exists(hparams.history_buffer_path):
        os.makedirs(hparams.history_buffer_path)

    # Set data load path
    DATA_DIR = os.getenv('PT_DATA_DIR')
    hparams.data_path = os.path.join(DATA_DIR, hparams.data_dir)  #e.g. $PT_DATA_DIR/asknav/
    hparams.img_features = os.path.join(DATA_DIR, 'img_features/ResNet-152-imagenet.tsv')

def save(path, model, optimizer, iter, best_metrics, train_env):
    '''save model checkpt'''

    ckpt = {
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'hparams'         : hparams,
            'iter'            : iter,
            'best_metrics'    : best_metrics,
            'data_idx'        : train_env.ix,
            'vocab'           : train_env.tokenizer.vocab
        }
    torch.save(ckpt, path)

def load(path, device):
    global hparams
    ckpt = torch.load(path, map_location=device)
    hparams = ckpt['hparams']

    # Overwrite hparams by args
    for flag in vars(args):
        value = getattr(args, flag)
        if value is not None:
            setattr(hparams, flag, value)

    set_path()
    return ckpt

def compute_ask_stats(traj):
    """
    compute stats related to asking mechanism
    (losses are not computed here)
    """

    total_steps = 0
    total_agent_ask = 0
    total_teacher_ask = 0
    agent_queries_per_ep = []
    ask_pred = []
    ask_true = []

    teacher_all_reasons = []
    ask_loss_str = ''

    # loop through one datapt at a time
    for i, t in enumerate(traj):
        assert len(t['agent_ask']) == len(t['teacher_ask'])
        end_step = len(t['agent_path'])
        pred = t['agent_ask'][:end_step]
        true = t['teacher_ask'][:end_step]

        total_steps += len(true)
        total_agent_ask += sum(x == AskAgent.ask_actions.index('ask') for x in pred)
        total_teacher_ask += sum(x == AskAgent.ask_actions.index('ask') for x in true)
        ask_pred.extend(pred)
        ask_true.extend(true)

        agent_queries_per_ep.append(sum(x == AskAgent.ask_actions.index('ask') for x in pred))
        teacher_reason = t['teacher_ask_reason'][:end_step]
        teacher_teacher_all_reasons.extend(teacher_reason)

    ask_loss_str += '\n *** ASK:'
    ask_loss_str += ' agent_queries_per_ep %.1f' % (sum(agent_queries_per_ep) / len(agent_queries_per_ep))
    ask_loss_str += ', agent_ratio %.3f' %  (total_agent_ask  / total_steps)
    ask_loss_str += ', teacher_ratio %.3f' % (total_teacher_ask / total_steps)
    ask_loss_str += ', A/P/R/F %.3f / %.3f / %.3f / %.3f' % (
                                            accuracy_score(ask_true, ask_pred),
                                            precision_score(ask_true, ask_pred),
                                            recall_score(ask_true, ask_pred),
                                            f1_score(ask_true, ask_pred))    

    ask_loss_str += '\n *** TEACHER ASK:'
    teacher_reason_counter = Counter(teacher_all_reasons)
    teacher_total_asks = sum(x != 'pass' and x != 'exceed' for x in teacher_all_reasons)
    ask_loss_str += ' ask %.3f, dont_ask %.3f, ' % (
            teacher_total_asks / len(teacher_all_reasons),
            (len(teacher_all_reasons) - teacher_total_asks) / len(teacher_all_reasons)
        )
    ask_loss_str += ', '.join(
        ['%s %.3f' % (k, teacher_reason_counter[k] / teacher_total_asks)
            for k in teacher_reason_counter.keys() if k not in ['pass', 'exceed']])

    return ask_loss_str    


def train(train_env, val_envs, agent, model, optimizer, start_iter, end_iter,
    best_metrics, eval_mode, explore_env=None):
    '''
    Main training loop. Loop through hparams.n_iters.
    - calls agent.train() which includes backprop
    - calls agent.test() which writes results to json files
    - print losses and metrics to stdout using `loss_str`
    '''
    start = time.time()

    SW = SummaryWriter(hparams.tensorboard_dir, flush_secs=30)
    # SW = SummaryWriter(os.path.join(hparams.tensorboard_dir, '' if hparams.plot_to_philly else 'main'), flush_secs=30)

    if not eval_mode:
        print('Training with with lr = %f' % optimizer.param_groups[0]['lr'])

    # Determine how agent should carry on to next timestep, teacher/argmax/sample
    train_feedback = { 'nav' : hparams.nav_feedback, 'ask' : hparams.ask_feedback, 'recover' : hparams.recover_feedback}
    test_feedback  = { 'nav' : 'argmax', 'ask' : 'argmax', 'recover' : 'argmax' }  

    loss_types = ["nav_losses", 'ask_losses', 'value_losses', 'recover_losses']

    # Set criteria for replacing the best model
    sr = 'success_rate'

    # Initialize loss string to be printed to stdout
    loss_str = ''

    for idx in range(start_iter, end_iter, hparams.log_every):

        # 1 iter = 1 batch of examples
        # 1 interval = default 1000 batches of examples
        interval = min(hparams.log_every, end_iter - idx)
        # iter index at the end of this interval
        iter = idx + interval 

        if eval_mode:
            loss_str = '\n * eval mode' 
        else:
            # Train for `interval` number of iterations
            # Calls rollout and backprop 
            # this returned traj only includes trajectories from the last 10 batches (to compute sample stats only)
            traj, time_report, nodes_time_report = agent.train(
                env=train_env, 
                optimizer=optimizer, 
                n_iters=interval, 
                feedback=train_feedback, 
                idx=idx,
                explore_env=explore_env)  # Debug TODO nodes_time_report
            SW.add_scalar('expert rollin - beta', agent.beta, iter)

            # Debug TODO
            print ("debug nodes_time_report here")
        
            # Report time for rollout and backprop
            for time_key, time_val in sorted(list(time_report.items()), key=lambda x: x[1], reverse=True):
                print ("Train {} time = {}".format(time_key, time_val)) 

            # Report per `interval` agent rollout losses
            # Main training loss -- summed all loss types
            train_losses = np.array(agent.losses)
            assert len(train_losses) <= interval  # one scalar per batch (i.e. iter)
            train_loss_avg = np.average(train_losses)  # across 1000 batches (i.e. iters)  
            loss_str = '\n * train loss: %.4f' % train_loss_avg
            SW.add_scalar('train - all losses', train_loss_avg, iter)

            # sanity check all necessary loss types exist
            if hparams.navigation_objective == 'action_imitation':
                assert hasattr(agent, "nav_losses") and len(agent.nav_losses) <= interval
            if hparams.navigation_objective == 'value_estimation':
                assert hasattr(agent, "value_losses") and len(agent.value_losses) <= interval
            if hparams.uncertainty_handling != 'no_ask':
                assert hasattr(agent, "ask_losses") and len(agent.ask_losses) <= interval
            if hparams.recovery_strategy != 'no_recovery':
                assert hasattr(agent, "recover_losses") and len(agent.recover_losses) <= interval        

            # Log individual training loss types (navigation, ask, value, recovery)
            for loss_type in loss_types:
                if hasattr(agent, loss_type):
                    # e.g. agent.nav_losses, agent.ask_losses, agent.value_losses, agent.recover_losses
                    train_loss_type_avg = np.average(np.array(getattr(agent ,loss_type)))
                    # convert "ask_losses" to "ask loss"
                    typ_str = " ".join(loss_type[:-2].split("_"))  
                    loss_str += ', %s: %.4f' % (typ_str, train_loss_type_avg)
                    if loss_type == 'ask_losses':
                        loss_str += compute_ask_stats(traj)
                    SW.add_scalar("train - " + typ_str, train_loss_type_avg, iter)
            
        metrics = defaultdict(dict)  # store success rates, ... etc
        should_save_ckpt = []  # save if model is the best for seen, unseen or combined

        # Run validation
        for env_name, (env, evaluator) in val_envs.items():

            # Get validation losses under the same conditions as training
            # rollout again -- time-consuming
            agent.test(
                env=env, 
                feedback=train_feedback, 
                explore_env=explore_env, 
                use_dropout=True, 
                allow_cheat=True)

            # Report per dataset agent rollout losses
            # Main validation loss -- summed all loss types
            val_loss_avg = np.average(agent.losses)
            loss_str += '\n * %s loss: %.4f' % (env_name, val_loss_avg)
            SW.add_scalar('%s - all losses' % env_name, val_loss_avg, iter)

            # Individual validation loss types (navigation, ask, value, recovery)
            for loss_type in loss_types:
                if hasattr(agent, loss_type):
                    # e.g. agent.nav_losses, agent.ask_losses, agent.value_losses, agent.recover_losses
                    val_loss_type_avg = np.average(np.array(getattr(agent ,loss_type)))
                    # convert "ask_losses" to "ask loss" for logging
                    typ_str = " ".join(loss_type[:-2].split("_"))  
                    loss_str += ', %s: %.4f' % (typ_str, val_loss_type_avg)
                    SW.add_scalar("{} - {}".format(env_name, typ_str), val_loss_type_avg, iter)
            
            # Get validation distance from goal under test evaluation conditions
            # rollout again -- time-consuming
            traj = agent.test(env=env, 
            feedback=test_feedback, 
            explore_env=explore_env, 
            use_dropout=False, 
            allow_cheat=False)

            # Write the validation results out to json file
            # Compute in both train and test modes
            agent.results_path = hparams.load_path.replace('.ckpt', '_') + env_name + '_for_eval.json'
            agent.write_results() # doesn't write is_success results yet
            score_summary, _, is_success = evaluator.score(agent.results_path)

            # Add success 1/0 to each traj
            # Compute only in eval mode (e.g. on test_seen or test_unseen data)
            if eval_mode:
                agent.results_path = hparams.load_path.replace('.ckpt', '_') + env_name + '_for_eval_complete.json'
                agent.add_is_success(is_success)
                print('Save result with is_success metric to', agent.results_path)
                agent.write_results()
            
            # compute success metrics and print in terminal 
            for metric, val in score_summary.items():
                # oracle_rate : if along the path, agent ever came close enough to goal
                # nav_error : distance between final pos and goal
                # length : average traj len
                # steps : average traj steps
                if metric in ['success_rate', 'oracle_rate', 'room_success_rate', 
                    'nav_error', 'length', 'steps']:
                    metrics[metric][env_name] = (val, len(traj))
                    SW.add_scalar("{} - {}".format(env_name, metric), val, iter)

                if metric in ['success_rate', 'oracle_rate', 'room_success_rate']:
                    loss_str += ', %s: %.3f' % (metric, val) 

            # print validation success metrics to terminal
            loss_str += '\n *** OTHER METRICS: '
            loss_str += '%s: %.2f' % ('nav_error', score_summary['nav_error'])
            loss_str += ', %s: %.2f' % ('oracle_error', score_summary['oracle_error'])
            loss_str += ', %s: %.2f' % ('length', score_summary['length'])
            loss_str += ', %s: %.2f' % ('steps', score_summary['steps'])
            if hparams.uncertainty_handling != 'no_ask':
                loss_str += compute_ask_stats(traj)

            # if success rate in this interval is better than the last, save the checkpt
            if not eval_mode and metrics[sr][env_name][0] > best_metrics[env_name]:
                should_save_ckpt.append(env_name)
                best_metrics[env_name] = metrics[sr][env_name][0]
                print('best %s success rate %.3f' % (env_name, best_metrics[env_name]))                                       

        # save the best combined model if we are not running eval
        if not eval_mode:
            combined_metric = (
                metrics[sr]['val_seen'][0]   * metrics[sr]['val_seen'][1] + \
                metrics[sr]['val_unseen'][0] * metrics[sr]['val_unseen'][1]) / \
                (metrics[sr]['val_seen'][1]  + metrics[sr]['val_unseen'][1])
            if combined_metric > best_metrics['combined']:
                should_save_ckpt.append('combined')
                best_metrics['combined'] = combined_metric

        # log time for each interval
        print('%s (%d %d%%) %s' % (timeSince(start, float(iter)/end_iter),
            iter, float(iter)/end_iter*100, loss_str))       

        if eval_mode:
            # Simply return the metrics
            res = defaultdict(dict)
            for metric in metrics:
                for k, v in metrics[metric].items():
                    res[metric][k] = v[0]
            return res 
        else:
            # Learning rate decay
            if hparams.lr_decay_rate and combined_metric < best_metrics['combined'] \
                and iter >= hparams.start_lr_decay and iter % hparams.decay_lr_every == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= hparams.lr_decay_rate
                    print('New learning rate %f' % param_group['lr'])

            # Save lastest model
            if iter == end_iter or iter % hparams.save_every == 0:
                should_save_ckpt.append('last')

            for env_name in should_save_ckpt:
                save_path = os.path.join(hparams.exp_dir,
                    '%s_%s.ckpt' % (hparams.model_prefix, env_name))
                save(save_path, model, optimizer, iter, best_metrics, train_env)
                print("Saved %s model to %s" % (env_name, save_path))

            # Save latest history buffer
            # save_buffer_start_time = time.time()
            # if (iter == end_iter or iter % hparams.save_every == 0) and \
            #     hparams.navigation_objective == 'value_estimation':
            #     agent.history_buffer.save_buffer(hparams.history_buffer_path)
            # print("Saving history buffer time = {}".format(time.time() - save_buffer_start_time))

            # log time again after saving
            print('%s (%d %d%%) %s' % (timeSince(start, float(iter)/end_iter),
                iter, float(iter)/end_iter*100, loss_str)) 

        # progress per training job on philly
        print("PROGRESS: {}%".format(float(iter)/end_iter*100))

    return None

def setup(seed=None):
    '''
    Set up seeds.
    Set up natural language vocabulary.
    '''
    if seed is not None:
        hparams.seed = seed
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    # Check for vocabs
    train_vocab_path = os.path.join(hparams.data_path, 'train_vocab.txt')
    if not os.path.exists(train_vocab_path):
        write_vocab(build_vocab(
                    hparams.data_path,
                    splits=['train'],
                    min_count=hparams.min_word_count,
                    max_length=hparams.max_input_length,
                    split_by_spaces=hparams.split_by_spaces,
                    prefix='noroom' if hasattr(hparams, 'no_room') and
                           hparams.no_room else 'asknav',
                    suffix=hparams.data_suffix if hasattr(hparams, 'data_suffix') else ''),
            train_vocab_path)     

def train_val(device, seed=None):
    ''' 
    Train on the training set, and validate on seen and unseen splits.
    - load last model 
    - define Env(s) for training and validation.
    - setup language vocab
    - specify choice of agent
    - specify choice of seq2seq model

    Returns:
        - calls main train() i.e. the main training loop
    '''

    # Resume from lastest checkpoint (if any)
    if os.path.exists(hparams.load_path):
        print('Load model from %s' % hparams.load_path)
        ckpt = load(hparams.load_path, device)
        start_iter = ckpt['iter']
    else:
        if hasattr(args, 'load_path') and hasattr(args, 'eval_only') and args.eval_only:
            sys.exit('load_path %s does not exist!' % hparams.load_path)
        ckpt = None
        start_iter = 0
    end_iter = hparams.n_iters

    # Setup seed and read vocab
    setup(seed=seed)

    # Set vocabulary
    train_vocab_path = os.path.join(hparams.data_path, 'train_vocab.txt')
    if hasattr(hparams, 'external_main_vocab') and hparams.external_main_vocab:
        train_vocab_path = hparams.external_main_vocab 
    # Optionally add subgoal vocabulary
    if 'verbal' in hparams.ask_advisor:
        subgoal_vocab_path = os.path.join(hparams.data_path, hparams.subgoal_vocab)
        vocab = read_vocab([train_vocab_path, subgoal_vocab_path])
    else:
        vocab = read_vocab([train_vocab_path])
    # passed into env which encodes sentences to token IDs
    tok = Tokenizer(vocab=vocab, encoding_length=hparams.max_input_length)

    # Create a training environment
    train_env = VNLABatch(hparams, split='train', tokenizer=tok)

    # Create validation environments
    # Define validation (val) environments
    val_splits = ['val_seen', 'val_unseen']
    # Define validation (test) environments
    eval_mode = hasattr(hparams, 'eval_only') and hparams.eval_only
    if eval_mode:
        if '_unseen' in hparams.load_path:
            val_splits = ['test_unseen']
        if '_seen' in hparams.load_path:
            val_splits = ['test_seen']
        end_iter = start_iter + hparams.log_every 
    # Create the seen and unseen environments
    val_envs = { split: (VNLABatch(hparams, split=split, tokenizer=tok,
        from_train_env=train_env, traj_len_estimates=train_env.traj_len_estimates),
        Evaluation(hparams, [split], hparams.data_path)) for split in val_splits}

    # Build model 
    # -- nav action imitation policy or cost-to-go value estimator
    # -- single-head or multi-head
    if hparams.navigation_objective == 'action_imitation' and not hparams.bootstrap:
        print ("Action Imitation model: Navigation Policy with single-head and continous LSTM decoder") 
        model = AttentionSeq2SeqContinuousModel(len(vocab), hparams, device)
    elif hparams.navigation_objective == 'action_imitation' and hparams.bootstrap:
        print ("Action Imitation model: Navigation Policy with mutli-head and continous LSTM decoder") 
        # model = AttentionSeq2SeqContinuousMultiHeadModel(len(vocab), hparams, device)
        raise NotImplementedError
    elif hparams.navigation_objective == 'value_estimation' and not hparams.bootstrap:
        print ("Value Estimation model: Cost-to-go Estimator with single-head and recent-frames LSTM decoder") 
        model = AttentionSeq2SeqFramesModel(len(vocab), hparams, device)
    elif hparams.navigation_objective == 'value_estimation' and hparams.bootstrap:
        print ("Value Estimation model: Cost-to-go Estimator with mutli-head and recent-frames LSTM decoder")
        # model = AttentionSeq2SeqFramesMultiHeadModel(len(vocab), hparams, device)
        raise NotImplementedError
    else:
        raise ValueError('model definition is not clear. check navigation_objective argument in hparams config')

    optimizer = optim.Adam(model.parameters(), lr=hparams.lr,
        weight_decay=hparams.weight_decay)
    
    # used to determine if we should checkpt the model
    best_metrics = { 'val_seen'  : -1,
                     'val_unseen': -1,
                     'combined'  : -1 }

    # Load model parameters from a checkpoint (if any)
    if ckpt is not None:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        best_metrics = ckpt['best_metrics']
        train_env.ix = ckpt['data_idx']   

    print('')
    pprint(vars(hparams), width=1)
    print('')
    print(model)

    # Choose agent
    # -- nav action imitation policy or cost-to-go value estimator
    # -- no ask or learned ask
    # -- no recovery or learned recovery    
    if hparams.navigation_objective == 'action_imitation':

        if hparams.uncertainty_handling == 'no_ask':
            print ("Action Imitation Agent: Navigation Policy, No Asking")
            agent = ActionImitationNoAskAgent(model, hparams, device)

        elif hparams.uncertainty_handling == 'learned_ask_flag':
            print ("Action Imitation Agent: Navigation Policy, Learned Asking")
            agent = ActionImitationVerbalAskAgent(model, hparams, device)

        else:
            raise ValueError

        explore_env = None

    elif hparams.navigation_objective == 'value_estimation':

        if hparams.uncertainty_handling == 'no_ask' and hparams.recovery_strategy == 'no_recovery':
            print ("Aggrevate Agent: Cost-to-go Estimator, No Asking, No Recovery")
            agent = ValueEstimationNoAskNoRecoveryAgent(model, hparams, device) 

        elif hparams.uncertainty_handling == 'learned_ask_flag' and hparams.recovery_strategy == 'no_recovery':
            print ("Aggrevate Agent: Cost-to-go Estimator, Learned Asking, No Recovery")
            #agent = ValueEstimationVerbalAskNoRecoveryAgent(model, hparams, device)
            raise NotImplementedError

        elif hparams.uncertainty_handling == 'no_ask' and hparams.recovery_strategy == 'learned_recovery':
            print ("Aggrevate Agent: Cost-to-go Estimator, No Asking, Learned Recovery")
            #agent = ValueEstimationNoAskRecoveryAgent(model, hparams, device)
            raise NotImplementedError

        elif hparams.uncertainty_handling == 'learned_ask_flag' and hparams.recovery_strategy == 'learned_recovery':
            print ("Aggrevate Agent: Cost-to-go Estimator, Learned Asking, Learned Recovery")
            #agent = ValueEstimationVerbalAskRecoveryAgent(model, hparams, device)
            raise NotImplementedError
        
        else:
            raise ValueError

        explore_env = VNLAExplorationBatch(agent.nav_actions, agent.env_actions, obs=None, batch_size=hparams.batch_size)

        # Load history buffer
        if os.path.exists(hparams.load_path):
            print('Load history buffer from %s' % hparams.history_buffer_path)
            agent.history_buffer.load_buffer(hparams.history_buffer_path)

    else:
        raise ValueError('agent definition is not clear. check navigation_objective')

    # Train
    return train(train_env, val_envs, agent, model, optimizer, start_iter, end_iter,
          best_metrics, eval_mode, explore_env)


if __name__ == "__main__":

    parser = make_parser()
    args = parser.parse_args()

    # Read configuration from a json file
    with open(args.config_file) as f:
        hparams = Namespace(**json.load(f))  

    # Overwrite hparams by args
    for flag in vars(args):
        value = getattr(args, flag)
        if value is not None:
            setattr(hparams, flag, value)      

    set_path()

    device = torch.device('cuda')
    print ("CUDA device count = {}".format(torch.cuda.device_count()))

    if hasattr(hparams, 'multi_seed_eval') and hparams.multi_seed_eval:
        hparams.eval_only = 1
        seeds = [123, 435, 6757, 867, 983]
        metrics = defaultdict(lambda: defaultdict(list))
        for seed in seeds:
            this_metrics = train_val(seed=seed)
            for metric in this_metrics:
                for k, v in this_metrics[metric].items():
                    if 'rate' in metric:
                        v *= 100
                    metrics[metric][k].append(v)
        for metric in metrics:
            for k, v in metrics[metric].items():
                print('%s %s: %.2f %.2f' % (metric, k, np.average(v), stats.sem(v) * 1.95))
    else:
        # Train
        train_val(device)        


# TURN OFF WHEN NOT USING VS code debugger------------------------------------------------------------------------------
hparams = None
args = None

def vs_code_debug(args_temp):

    global hparams
    global args

    parser = make_parser()
    args = parser.parse_args()

    # Read configuration from a json file
    with open(args_temp["config_file"]) as f:
        hparams = Namespace(**json.load(f)) 

    # Overwrite hparams by args
    for flag in vars(args):
        value = getattr(args, flag)
        if value is not None:
            setattr(hparams, flag, value)   

    # Overwrite hparams by args_temp
    for flag in args_temp:
        value = args_temp[flag]
        setattr(hparams, flag, value)

    set_path()
    print ("set_path() is done")

    device = torch.device('cuda')

    print ("CUDA device count = {}".format(torch.cuda.device_count()))

    if hasattr(hparams, 'multi_seed_eval') and hparams.multi_seed_eval:
        hparams.eval_only = 1
        seeds = [123, 435, 6757, 867, 983]
        metrics = defaultdict(lambda: defaultdict(list))
        for seed in seeds:
            this_metrics = train_val(seed=seed)
            for metric in this_metrics:
                for k, v in this_metrics[metric].items():
                    if 'rate' in metric:
                        v *= 100
                    metrics[metric][k].append(v)
        for metric in metrics:
            for k, v in metrics[metric].items():
                print('%s %s: %.2f %.2f' % (metric, k, np.average(v), stats.sem(v) * 1.95))
    else:
        # Train
        train_val(device)
# ------------------------------------------------------------------------------