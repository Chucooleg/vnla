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

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince
from env import VNLABatch
from model import AttentionSeq2SeqModel
from ask_agent import AskAgent
from verbal_ask_agent import VerbalAskAgent
from tensorboardX import SummaryWriter

from eval import Evaluation
from oracle import *
from flags import make_parser


def set_path():

    # Set general output dir
    hparams.exp_dir = os.getenv('PT_EXP_DIR')

    hparams.model_prefix = '%s_nav_%s_ask_%s' % (hparams.exp_name,
        hparams.nav_feedback, hparams.ask_feedback)

    # Set tensorboard log dir
    if hparams.local_run:
        hparams.tensorboard_dir = hparams.exp_dir
    else:
        if hparams.plot_to_philly:
            hparams.tensorboard_dir = os.environ.get('PHILLY_LOG_DIRECTORY', '.')
        else:
            hparams.tensorboard_dir = os.environ.get('PT_TENSORBOARD_DIR', '.')
    print ("tensorboard dir = {}".format(hparams.tensorboard_dir))

    # Set model load path
    hparams.load_path = hparams.load_path if hasattr(hparams, 'load_path') and \
        hparams.load_path is not None else \
        os.path.join(hparams.exp_dir, '%s_last.ckpt' % hparams.model_prefix)

    # Set data load path
    DATA_DIR = os.getenv('PT_DATA_DIR')
    hparams.data_path = os.path.join(DATA_DIR, hparams.data_dir)  #e.g. $PT_DATA_DIR/asknav/
    hparams.img_features = os.path.join(DATA_DIR, hparams.img_features)
    # hparams.img_features = os.path.join(DATA_DIR, 'img_features/ResNet-152-imagenet.tsv')

    # semantics update!
    hparams.room_types_path = os.path.join(DATA_DIR, hparams.room_types_path)

def save(path, model, optimizer, iter, best_metrics, train_env):
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
    total_steps = 0
    total_agent_ask = 0
    total_teacher_ask = 0
    queries_per_ep = []
    ask_pred = []
    ask_true = []

    all_reasons = []
    loss_str = ''

    for i, t in enumerate(traj):
        assert len(t['agent_ask']) == len(t['teacher_ask'])

        end_step = len(t['agent_path'])

        pred = t['agent_ask'][:end_step]
        true = t['teacher_ask'][:end_step]

        total_steps += len(true)
        total_agent_ask  += sum(x == AskAgent.ask_actions.index('ask') for x in pred)
        total_teacher_ask += sum(x == AskAgent.ask_actions.index('ask') for x in true)
        ask_pred.extend(pred)
        ask_true.extend(true)

        queries_per_ep.append(sum(x == AskAgent.ask_actions.index('ask') for x in pred))
        teacher_reason = t['teacher_ask_reason'][:end_step]
        all_reasons.extend(teacher_reason)

    loss_str += '\n *** ASK:'
    loss_str += ' queries_per_ep %.1f' % (sum(queries_per_ep) / len(queries_per_ep))
    loss_str += ', agent_ratio %.3f' %  (total_agent_ask  / total_steps)
    loss_str += ', teacher_ratio %.3f' % (total_teacher_ask / total_steps)
    loss_str += ', A/P/R/F %.3f / %.3f / %.3f / %.3f' % (
                                            accuracy_score(ask_true, ask_pred),
                                            precision_score(ask_true, ask_pred),
                                            recall_score(ask_true, ask_pred),
                                            f1_score(ask_true, ask_pred))

    loss_str += '\n *** TEACHER ASK:'
    reason_counter = Counter(all_reasons)
    total_asks = sum(x != 'pass' and x != 'exceed' for x in all_reasons)
    loss_str += ' ask %.3f, dont_ask %.3f, ' % (
            total_asks / len(all_reasons),
            (len(all_reasons) - total_asks) / len(all_reasons)
        )
    loss_str += ', '.join(
        ['%s %.3f' % (k, reason_counter[k] / total_asks)
            for k in reason_counter.keys() if k not in ['pass', 'exceed']])

    return loss_str

def train(train_env, val_envs, agent, model, optimizer, start_iter, end_iter,
    best_metrics, eval_mode):

    if not eval_mode:
        print('Training with with lr = %f' % optimizer.param_groups[0]['lr'])

    train_feedback = { 'nav' : hparams.nav_feedback, 'ask' : hparams.ask_feedback }
    test_feedback  = { 'nav' : 'argmax', 'ask' : 'argmax' }

    start = time.time()
    sr = 'success_rate'

    SW = SummaryWriter(hparams.tensorboard_dir, flush_secs=30)
    # SW = SummaryWriter(os.environ.get('PHILLY_LOG_DIRECTORY', '.'), flush_secs=30) # pull from browser

    for idx in range(start_iter, end_iter, hparams.log_every):
        # An iter is a batch

        interval = min(hparams.log_every, end_iter - idx) # An interval is a number of batches
        iter = idx + interval # iter index at the end of this interval

        # Train for log_every iterations
        if eval_mode:
            loss_str = '\n * eval mode'
        else:
            traj, time_keep = agent.train(train_env, optimizer, interval, train_feedback, idx)
            
            # timing for different processes in rollout and loss backward
            for key in sorted(time_keep.keys()):
                print ("{} time = {}".format(key, time_keep[key]))

            train_losses = np.array(agent.losses)
            assert len(train_losses) == interval
            train_loss_avg = np.average(train_losses)

            loss_str = '\n * train loss: %.4f' % train_loss_avg
            train_nav_loss_avg = np.average(np.array(agent.nav_losses))
            train_ask_loss_avg = np.average(np.array(agent.ask_losses))
            loss_str += ', nav loss: %.4f' % train_nav_loss_avg
            loss_str += ', ask loss: %.4f' % train_ask_loss_avg
            loss_str += compute_ask_stats(traj)

            SW.add_scalar('train loss per {} iters'.format(hparams.log_every), train_loss_avg, iter)
            SW.add_scalar('train nav loss per {} iters'.format(hparams.log_every), train_nav_loss_avg, iter)
            SW.add_scalar('train ask loss per {} iters'.format(hparams.log_every), train_ask_loss_avg, iter)

        metrics = defaultdict(dict)
        should_save_ckpt = []

        # Run validation
        for env_name, (env, evaluator) in val_envs.items():
            # Get validation loss under the same conditions as training
            agent.test(env, train_feedback, use_dropout=True, allow_cheat=True)
            val_loss_avg = np.average(agent.losses)
            loss_str += '\n * %s loss: %.4f' % (env_name, val_loss_avg)
            val_nav_loss_avg = np.average(agent.nav_losses)
            loss_str += ', nav loss: %.4f' % val_nav_loss_avg
            val_ask_loss_avg = np.average(agent.ask_losses)
            loss_str += ', ask loss: %.4f' % val_ask_loss_avg

            SW.add_scalar('{} loss per {} iters'.format(env_name, hparams.log_every), val_loss_avg, iter)
            SW.add_scalar('{} nav loss per {} iters'.format(env_name, hparams.log_every), val_nav_loss_avg, iter)
            SW.add_scalar('{} ask loss per {} iters'.format(env_name, hparams.log_every), val_ask_loss_avg, iter)

            # Get validation distance from goal under test evaluation conditions
            traj = agent.test(env, test_feedback, use_dropout=False, allow_cheat=False)

            agent.results_path = os.path.join(hparams.exp_dir,
                '%s_%s_for_eval.json' % (hparams.model_prefix, env_name))
            agent.write_results(traj)
            score_summary, _, is_success = evaluator.score(agent.results_path)

            if eval_mode:
                agent.results_path = hparams.load_path.replace('ckpt', '') + env_name + '.json'
                agent.add_is_success(is_success)
                print('Save result to', agent.results_path)
                agent.write_results(traj)

            for metric, val in score_summary.items():
                if metric in ['success_rate', 'oracle_rate', 'room_success_rate',
                    'nav_error', 'length', 'steps']:
                    metrics[metric][env_name] = (val, len(traj))
                if metric in ['success_rate', 'oracle_rate', 'room_success_rate']:
                    loss_str += ', %s: %.3f' % (metric, val)
                    SW.add_scalar('{} {} steps per {} iters'.format(env_name, metric, hparams.log_every), val, iter)

            loss_str += '\n *** OTHER METRICS: '
            loss_str += '%s: %.2f' % ('nav_error', score_summary['nav_error'])
            loss_str += ', %s: %.2f' % ('oracle_error', score_summary['oracle_error'])
            loss_str += ', %s: %.2f' % ('length', score_summary['length'])
            loss_str += ', %s: %.2f' % ('steps', score_summary['steps'])
            loss_str += compute_ask_stats(traj)

            SW.add_scalar('{} nav_error per {} iters'.format(env_name, hparams.log_every), score_summary['nav_error'], iter)
            SW.add_scalar('{} oracle_error per {} iters'.format(env_name, hparams.log_every), score_summary['oracle_error'], iter)
            SW.add_scalar('{} length per {} iters'.format(env_name, hparams.log_every), score_summary['length'], iter)
            SW.add_scalar('{} steps per {} iters'.format(env_name, hparams.log_every), score_summary['steps'], iter)

            if not eval_mode and metrics[sr][env_name][0] > best_metrics[env_name]:
                should_save_ckpt.append(env_name)
                best_metrics[env_name] = metrics[sr][env_name][0]
                print('best %s success rate %.3f' % (env_name, best_metrics[env_name]))

        if not eval_mode:
            combined_metric = (
                metrics[sr]['val_seen'][0]   * metrics[sr]['val_seen'][1] + \
                metrics[sr]['val_unseen'][0] * metrics[sr]['val_unseen'][1]) / \
                (metrics[sr]['val_seen'][1]  + metrics[sr]['val_unseen'][1])
            if combined_metric > best_metrics['combined']:
                should_save_ckpt.append('combined')
                best_metrics['combined'] = combined_metric
                print('best combined success rate %.3f' % combined_metric)

        print('%s (%d %d%%) %s' % (timeSince(start, float(iter)/end_iter),
            iter, float(iter)/end_iter*100, loss_str))

        if eval_mode:
            res = defaultdict(dict)
            for metric in metrics:
                for k, v in metrics[metric].items():
                    res[metric][k] = v[0]
            return res

        if not eval_mode:
            # Learning rate decay
            if hparams.lr_decay_rate and combined_metric < best_metrics['combined'] \
                and iter >= hparams.start_lr_decay and iter % hparams.decay_lr_every == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= hparams.lr_decay_rate
                    print('New learning rate %f' % param_group['lr'])

            # Save lastest model?
            if iter == end_iter or iter % hparams.save_every == 0:
                should_save_ckpt.append('last')

            for env_name in should_save_ckpt:
                save_path = os.path.join(hparams.exp_dir,
                    '%s_%s.ckpt' % (hparams.model_prefix, env_name))
                save(save_path, model, optimizer, iter, best_metrics, train_env)
                print("Saved %s model to %s" % (env_name, save_path))
        
        # progress per training job on philly
        print("PROGRESS: {}%".format(float(iter)/end_iter*100))

    return None

def setup(seed=None):

    if seed is not None:
        hparams.seed = seed
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
                           hparams.no_room else 'asknav'),
            train_vocab_path)

def train_val(device, seed=None):
    ''' Train on the training set, and validate on seen and unseen splits. '''

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

    train_vocab_path = os.path.join(hparams.data_path, 'train_vocab.txt')
    if hasattr(hparams, 'external_main_vocab') and hparams.external_main_vocab:
        train_vocab_path = hparams.external_main_vocab

    if 'verbal' in hparams.advisor:
        subgoal_vocab_path = os.path.join(hparams.data_path, hparams.subgoal_vocab)
        vocab = read_vocab([train_vocab_path, subgoal_vocab_path])
    else:
        vocab = read_vocab([train_vocab_path])
    tok = Tokenizer(vocab=vocab, encoding_length=hparams.max_input_length)

    # Create a training environment
    train_env = VNLABatch(hparams, split='train', tokenizer=tok)

    # Create validation environments
    val_splits = ['val_seen', 'val_unseen']
    eval_mode = hasattr(hparams, 'eval_only') and hparams.eval_only
    if eval_mode:
        if '_unseen' in hparams.load_path:
            val_splits = ['test_unseen']
        if '_seen' in hparams.load_path:
            val_splits = ['test_seen']
        end_iter = start_iter + hparams.log_every

    val_envs = { split: (VNLABatch(hparams, split=split, tokenizer=tok,
        from_train_env=train_env, traj_len_estimates=train_env.traj_len_estimates),
        Evaluation(hparams, [split], hparams.data_path)) for split in val_splits}

    # Build models
    model = AttentionSeq2SeqModel(len(vocab), hparams, device)

    optimizer = optim.Adam(model.parameters(), lr=hparams.lr,
        weight_decay=hparams.weight_decay)

    print ("device count :".format(torch.cuda.device_count()))

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

    # semantics update!
    with open(hparams.room_types_path, "r") as f:
        room_types = f.read().split('\n')[:-1]
    print ('train room types \n', room_types)

    # Initialize agent
    if 'verbal' in hparams.advisor:
        # semantics update!
        agent = VerbalAskAgent(model, hparams, room_types, device)
    elif hparams.advisor == 'direct':
        # semantics update!
        agent = AskAgent(model, hparams, room_types, device)

    # Train
    return train(train_env, val_envs, agent, model, optimizer, start_iter, end_iter,
          best_metrics, eval_mode)


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
    
    # with torch.cuda.device(hparams.device_id):
    # with torch.cuda.device(device):
    # Multi-seed evaluation
    if hasattr(hparams, 'multi_seed_eval') and hparams.multi_seed_eval:
        args.eval_only = 1
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