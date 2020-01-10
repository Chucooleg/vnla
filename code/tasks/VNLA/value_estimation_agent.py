# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import division

import json
import os
import sys
import random
import time
import copy
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.distributions as D
from torch import optim
import torch.nn.functional as F

from utils import padding_idx
from nav_agent import NavigationAgent
from history_buffer import HistoryBuffer
from oracle import make_oracle
from env import EnvBatch


class ValueEstimationAgent(NavigationAgent):
    """
    Handle hparams and constraints that are applied to value estimation agents, w/o asking, w/o recovery.
    """

    # NOTE Add recovery actions here
    ask_actions = ['dont_ask', 'ask', '<start>', '<ignore>']
    # inherited nav_actions and env_actions
    
    def __init__(self, model, hparams, device):
        super(ValueEstimationAgent, self).__init__(model, hparams, device)

        self.batch_size = hparams.batch_size

        self.value_criterion = nn.SmoothL1Loss(reduction='mean') 

        # Oracle
        self.value_teacher = make_oracle('frontier_shortest', self.nav_actions, self.env_actions)

        # Explore pano sphere
        self.max_macro_action_seq_len = 8

        # Number of images in a pano sphere
        self.num_viewIndex = 36

        # Action tuple size
        self.env_action_tup_size = 3

        # Initialize History Buffer
        self.history_buffer = HistoryBuffer(hparams)
        self.min_history_to_learn = hparams.min_history_to_learn

        # Initialize q-value threshold for agent to determine if its path should 'end'
        self.end_q_val_threshold = hparams.end_q_val_threshold

        # Set expert rollin prob, decayed in training
        self.beta = 1.0
        self.start_beta_decay = hparams.start_beta_decay
        self.decay_beta_every = hparams.decay_beta_every
        self.beta_decay_rate = hparams.beta_decay_rate

        # Initialize loss, accumulated in subclasses
        self.loss = 0.0

        # Only for computing validation losses under training conditions
        self.allow_cheat = False  
        # Compute losses if not eval (i.e. training)
        self.is_eval = False

    def make_tr_instructions_by_t(self, tr_key_pairs, tr_timesteps):
        '''
        Extract language instructions from history buffer
        Arguments:
            tr_key_pairs: list of len(batch_size) tuples, each (instr_id, global_iter_idx).
            tr_timesteps: list of len(batch_size) integers, each indicate sampled timestep
        '''
        # Extract language instruction and convert to token IDs
        seq_tensor = np.array([self.env.encode(self.history_buffer[instr_iter_key]['instruction'][timestep]) \
            for (instr_iter_key, timestep) in zip(tr_key_pairs, tr_timesteps)])

        # Get length of seq before padding idx pos
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        assert sum([length <= 0 for length in seq_lengths]) == 0

        # Set max allowable seq length
        max_length = max(seq_lengths)
        assert max_length <= self.max_input_length  # from hparam

        # Convert seq inputs into torch tensors for encoder
        seq_tensor = torch.from_numpy(seq_tensor).long().to(self.device)[:, :max_length]
        seq_lengths = torch.from_numpy(seq_lengths).long().to(self.device)

        # Make sequence mask to account for padding
        # shape (batch_size, max_length), 1 for padded token positions
        mask = (seq_tensor == padding_idx)

        return seq_tensor, mask, seq_lengths

    def get_tr_variable_by_t(self, tr_key_pairs, tr_timesteps, var_name):
        '''
        Extract variable from history buffer by (instr_id, global_iter_idx), timestep and variable name. 

        Arguments:
            tr_key_pairs: list of len(batch_size) tuples, each (instr_id, global_iter_idx)
            tr_timesteps: list of len(batch_size) integers, each indicate sampled timestep
            var_name: string. name of variable in "action"/"feature"/"q_values_target"

        Returns: 
            (batch_size, corresponding variable size)
        '''
        assert var_name in ('action', 'feature', 'q_values_target')

        # Create container to hold training data
        variable_dim = self.history_buffer[tr_key_pairs[0]][var_name][tr_timesteps[0]].shape[0]
        # shape (batch_size, variable size)
        variable = torch.empty(len(tr_key_pairs), variable_dim, dtype=torch.float, device=self.device)

        # Fill container with data extracted from history buffer
        for i, (key_pair, timestep) in enumerate(zip(tr_key_pairs, tr_timesteps)):
            variable[i,:] = self.history_buffer[key_pair][var_name][timestep]

        return variable

    # def get_tr_end_target_by_t(self, tr_key_pairs, tr_timesteps):
    #     '''
    #     Extract variable from history buffer by (instr_id, global_iter_idx), timestep and variable name. 

    #     Arguments:
    #         tr_key_pairs: list of len(batch_size) tuples, each (instr_id, global_iter_idx)
    #         tr_timesteps: list of len(batch_size) integers, each indicate sampled timestep

    #     Returns: 
    #         (batch_size, self.max_macro_action_seq_len)
    #     '''
    #     # Create container to hold training data
    #     end_target_dim = self.max_macro_action_seq_len
    #     # shape (batch_size, self.max_macro_action_seq_len)
    #     end_target = torch.empty(len(tr_key_pairs), end_target_dim, dtype=torch.float, device=self.device)

    #     # Fill container with data extracted from history buffer
    #     for i, (key_pair, timestep) in enumerate(zip(tr_key_pairs, tr_timesteps)):
    #         # list/arr of int indices -- can have >1 reachable goal points
    #         end_target_ixs = self.history_buffer[key_pair]['end_target_indices'][timestep]
    #         # fill the mask
    #         end_target[i, end_target_ixs] = 1
    #     return end_target

    def get_tr_view_indexed_action_by_t(self, tr_key_pairs, tr_timesteps, view_ix):
        '''
        Extract macro-action sequence associated with a viewIndex(rotation) from history buffer by 
        (instr_id, global_iter_idx), timestep, variable name and view index.

        Arguments:
            tr_key_pairs: list of len(batch_size) tuples, each (instr_id, global_iter_idx)
            tr_timesteps: list of len(batch_size) integers, each indicate sampled timestep
            view_ix: integer [0-35] used to index panoramic images in the Matterport 3D simulator. 

        Returns: 
            (batch_size, max macro-action seq length)
        '''

        # Create container to hold training data
        action_dim = self.history_buffer[tr_key_pairs[0]]['viewix_actions_map'][tr_timesteps[0]][view_ix].shape[0]
        # shape (batch_size, max macro-action sequence length) 
        actions = torch.empty(len(tr_key_pairs), action_dim, dtype=torch.float, device=self.device)

        # Fill container with data extracted from history buffer
        for i, (key_pair, timestep) in enumerate(zip(tr_key_pairs, tr_timesteps)):
            actions[i,:] = self.history_buffer[key_pair]['viewix_actions_map'][timestep][view_ix]
        return actions

    def get_tr_view_indexed_mask_by_t(self, tr_key_pairs, tr_timesteps, view_ix):
        '''
        Extract variable associated with a viewIndex(rotation) from history buffer by 
        (instr_id, global_iter_idx), timestep, variable name and view index.

        Arguments:
            tr_key_pairs: list of len(batch_size) tuples, each (instr_id, global_iter_idx)
            tr_timesteps: list of len(batch_size) integers, each indicate sampled timestep
            view_ix: integer [0-35] used to index panoramic images in the Matterport 3D simulator. 

        Returns: 
            (batch_size,)
        '''
        # Create container to hold training data
        # shape (batch_size,) 
        mask = torch.zeros(len(tr_key_pairs), dtype=torch.uint8, device=self.device)

        # Fill container with data extracted from history buffer
        for i, (key_pair, timestep) in enumerate(zip(tr_key_pairs, tr_timesteps)):
            mask[i] = int(view_ix in self.history_buffer[key_pair]['view_index_mask_indices'][timestep])

        return mask

    # def get_tr_view_indexed_full_mask_by_t(self, tr_key_pairs, tr_timesteps, num_viewIndex):
    #     '''
    #     Extract variable associated with a viewIndex(rotation) from history buffer by 
    #     (instr_id, global_iter_idx), timestep and variable name.

    #     Arguments:
    #         tr_key_pairs: list of len(batch_size) tuples, each (instr_id, global_iter_idx)
    #         tr_timesteps: list of len(batch_size) integers, each indicate sampled timestep

    #     Returns: 
    #         (batch_size, agent.num_viewIndex)
    #     '''
    #     # Create container to hold training data
    #     # shape (batch_size, agent.num_viewIndex)
    #     mask = torch.zeros(len(tr_key_pairs, num_viewIndex), dtype=torch.uint8, device=self.device)

    #     # Fill container with data extracted from history buffer
    #     for i, (key_pair, timestep) in enumerate(zip(tr_key_pairs, tr_timesteps)):
    #         indices = self.history_buffer[key_pair]['view_index_mask_indices'][timestep]
    #         mask[i, indices] = 1
    #     return mask
        
    def get_tr_view_indexed_features_by_t(self, tr_key_pairs, tr_timesteps, view_ix):
        '''
        Extract image features associated with a viewIndex(rotation) from history buffer by 
        (instr_id, global_iter_idx), timestep, and view index.

        Arguments:
            tr_key_pairs: list of len(batch_size) tuples, each (instr_id, global_iter_idx)
            tr_timesteps: list of len(batch_size) integers, each indicate sampled timestep
            view_ix: integer [0-35] used to index panoramic images in the Matterport 3D simulator. 

        Returns: 
            (batch_size, feature size)        
        '''
        # Create container to hold training data
        feature_dim = self.history_buffer[tr_key_pairs[0]]['feature'][tr_timesteps[0]].shape[0]
        # shape (batch_size, feature size) 
        features = np.empty((len(tr_key_pairs), feature_dim), dtype=np.float32) 

        # Fill container with data extracted from history buffer
        for i, (key_pair, timestep) in enumerate(zip(tr_key_pairs, tr_timesteps)):
            viewpoint_idx = self.history_buffer[key_pair]['viewpointIndex'][timestep]
            scan_id = self.history_buffer[key_pair]['scan']
            long_id = scan_id + '_' + viewpoint_idx
            features[i,:] = self.env.features[long_id][view_ix, :]

        return torch.from_numpy(features).to(self.device)

    def _setup(self, env, feedback):
        # Leave for subclasses to:
        # - setup nav, ask, recover feedback
        # - setup nav, ask, recover losses
        # - setup self.env
        # - add scans to teachers/advisors
        raise NotImplementedError('Subclasses are expected to implement _setup')

    def test(self, env, feedback, use_dropout=False, allow_cheat=False):
        ''' Evaluate once on each instruction in the current environment '''

        self.allow_cheat = allow_cheat
        self.is_eval = not allow_cheat

        # expert roll-in prob
        if self.allow_cheat:
            assert self.beta > 0.0  # always > 0.0 in exponential decay

        self._setup(env, feedback)
        if use_dropout:
            self.model.train()
        else:
            self.model.eval()
        return self.base_test(self, env)

    def train(self, env, optimizer, n_iters, feedback, idx):
        '''
        Train for a given number `n_iters` of iterations. 
        `n_iters` is set to hparams.log_every (default 1000) or remaining.
        '''

        self.is_eval = False

        # Set up self.env, feedback(teacher/argmax/learned)
        # Initialize losses, add scans to teachers/advisors
        self._setup(env, feedback)

        # Check expert roll-in prob
        assert self.beta > 0.0 # always > 0.0 in exponential decay

        # time report
        time_report = defaultdict(int)

        # List of 10*batch_size number of trajectory dictionaries
        last_traj = []

        # Loop through 1000 batches of datapts
        for itr in range(1, n_iters+1):

            # Global training iteration index
            global_iter_idx = idx + itr

            optimizer.zero_grad()

            # Rollout the agent
            # History is added to buffer within this rollout() call
            # See subclass implementation
            traj, iter_time_report_rollout = self.rollout(global_iter_idx)

            # Keep time for the rollout processes
            for key in iter_time_report_rollout.keys():
                time_report[key] += iter_time_report_rollout[key]

            # Save rollout trajectories from last 10 batches to compute sample stats (e.g. compute_ask_stats)
            if n_iters - itr <= 10:
                last_traj.extend(traj)

            # If history buffer has enough history to update policy
            if len(self.history_buffer) > self.min_history_to_learn:
                
                # Sample a minibatch from the history buffer
                start_time = time.time()
                sampled_training_batch = self.history_buffer.sample_minibatch(self.batch_size)
                time_report['sample_minibatch'] += time.time() - start_time

                # Make predictions on training batch
                # Compute self.loss, self.losses and self.value_losses on minibatch
                # See subclass implementation
                training_batch_results, iter_time_report_train = self.pred_training_batch(sampled_training_batch, global_iter_idx)

                # Keep time for the training batch processes
                for key in iter_time_report_train.keys():
                    time_report[key] += iter_time_report_train[key]

                # Compute gradients
                start_time = time.time()
                self.loss.backward()

                # NOTE add code to divide up gradients by 1/n_ensemble in the shared encoder if we are bootstrapping
                # if self.bootstrap:
                #     for param in self.model.encoder.parameters():
                #         if param.grad is not None:
                #             param.grad.data *= 1.0/float(self.n_ensemble)
                # NOTE add code here if gradient clipping

                # Update torch model params
                optimizer.step()
                time_report['backprop_time'] += time.time() - start_time

            # Exponential decay expert-rollin probability beta
            if global_iter_idx >= self.start_beta_decay and global_iter_idx % self.decay_beta_every == 0:
                self.beta *= (1 - self.beta_decay_rate)
                print('New expert roll-in probability %f' % self.beta)
            self.SW.add_scalar('beta per iter', self.beta, global_iter_idx)

        return last_traj, time_report        