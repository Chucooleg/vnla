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
# from history_buffer import HistoryBuffer
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
        self.tr_batch_size = hparams.train_batch_size

        if hparams.loss_function == 'l1':
            # Bootstrap DQN paper used Huber Loss
            # https://en.wikipedia.org/wiki/Huber_loss
            # if delta == 1 in Huber Loss
            # https://pytorch.org/docs/stable/nn.html?highlight=smooth%20l1#torch.nn.SmoothL1Loss
            self.loss_function_str = "l1"
            self.loss_function_ref_str = "l2"
            self.value_criterion = nn.SmoothL1Loss(reduction='sum')
            self.value_ref_criterion = nn.MSELoss(reduction='sum')
        elif hparams.loss_function == 'l2':
            self.loss_function_str = "l2"
            self.loss_function_ref_str = "l1"
            self.value_criterion = nn.MSELoss(reduction='sum')
            self.value_ref_criterion = nn.SmoothL1Loss(reduction='sum')
        self.norm_loss_by_dist = hparams.norm_loss_by_dist

        # Oracle
        self.value_teacher = make_oracle('frontier_shortest', self.nav_actions, self.env_actions)

        # Explore pano sphere
        self.max_macro_action_seq_len = 8

        # Number of images in a pano sphere
        self.num_viewIndex = 36

        # Action tuple size
        self.env_action_tup_size = 3

        # Initialize History Buffer
        # self.history_buffer = HistoryBuffer(hparams)
        self.min_history_to_learn = hparams.min_history_to_learn

        # Set expert rollin prob, decayed in training
        # self.beta = float(hparams.start_beta) if hasattr(hparams, "start_beta") else 1.0
        self.start_beta_decay = hparams.start_beta_decay
        self.decay_beta_every = hparams.decay_beta_every
        self.beta_decay_rate = hparams.beta_decay_rate

        # Initialize loss, accumulated in subclasses
        self.loss = 0.0

        # Only for computing validation losses under training conditions
        self.allow_cheat = False  
        # Compute losses if not eval (i.e. training)
        self.is_eval = False

    def normalize_loss_with_mask(self, batch_size, q_values_estimate_heads, q_values_target):

        assert q_values_estimate_heads.shape == (batch_size, self.num_viewIndex, self.n_ensemble)
        assert q_values_target.shape == (batch_size, self.num_viewIndex)

        # tensor shape (self.n_ensemble, )
        value_loss_heads = torch.empty(self.n_ensemble, dtype=torch.float, device=self.device)

        for h in range(self.n_ensemble):

            # Sample a mask -> 1 means the head can see the example
            # tensor shape (batch_size,)
            bootstrap_mask = torch.bernoulli(torch.ones(batch_size, dtype=torch.float, device=self.device) * self.bernoulli_probability)

            # Compute loss
            # tensor shape (batch_size, ), tensor shape (batch_size, )
            value_loss_h, not_ended_mask = self._normalize_loss_across_viewing_angles(batch_size, q_values_estimate_heads[:,:,h], q_values_target, ref=False)

            # Apply mask and Normalize
            # tensor scalar
            # import pdb; pdb.set_trace()
            value_loss_heads[h] = torch.sum(value_loss_h * bootstrap_mask * not_ended_mask) / torch.sum(bootstrap_mask * not_ended_mask)

        return torch.mean(value_loss_heads)

    def _normalize_loss_across_viewing_angles(self, batch_size, q_values_estimate, q_values_target, ref=False):
        '''
        Normalize value loss across 36 panoramic views per example
        batch_size : scalar
        q_values_estimate : torch tensor shape (batch_size, self.num_viewIndex)
        q_values_target   : torch tensor shape (batch_size, self.num_viewIndex)
        Returns:
            value_losses : torch tensor (batch_size, )
            not_ended_mask : torch tensor (batch_size, ). Use 1/0 to mark the tasks which has not `ended` their trajectory.
        '''
        value_criterion = self.value_ref_criterion if ref else self.value_criterion

        # tensor shape (batch_size, )
        value_losses = torch.empty(batch_size, dtype=torch.float, device=self.device)
        not_ended_mask = torch.empty(batch_size, dtype=torch.float, device=self.device)

        # dummy targets to compute normalized loss
        # tensor shape (self.num_viewIndex=36, )
        normalized_targets = torch.ones(self.num_viewIndex, dtype=torch.float, device=self.device) if self.norm_loss_by_dist else None

        # Average across 36 views, if the view is not masked
        # not_ended_count = 0
        for i in range(batch_size):
            visible_vertex_count = torch.sum(q_values_target[i] != 1e9)

            # if the task has not ended yet
            if visible_vertex_count:
                not_ended_mask[i] = 1
                # not_ended_count += 1

                # Compute loss WITH normalization by distance from current position to goal
                if self.norm_loss_by_dist:
                    # print('Applying loss normalization by distance-to-go for {} loss function..'.format(self.loss_function_str))
                    value_losses[i] = value_criterion(q_values_estimate[i]/q_values_target[i], normalized_targets) / visible_vertex_count
                # Compute loss WITHOUT normalization by distance
                else:
                    # print('Applying NO loss normalization by distance-to-go for {} loss function..'.format(self.loss_function_str))
                    value_losses[i] = value_criterion(q_values_estimate[i], q_values_target[i]) / visible_vertex_count

            # if the task has ended already
            else:
                not_ended_mask[i] = 0
                value_losses[i] = 0

        # return value_losses, not_ended_count
        return value_losses, not_ended_mask

    def normalize_loss(self, batch_size, q_values_estimate, q_values_target, ref=False):
        '''
        Normalize value loss first across panoramic views per example, 
        then across all examples in the batch, accounting for only tasks which has not 'ended' their episodes
        batch_size : scalar
        q_values_estimate : torch tensor shape (batch_size, self.num_viewIndex)
        q_values_target   : torch tensor shape (batch_size, self.num_viewIndex)        
        '''
        assert q_values_estimate.shape == q_values_target.shape
        # average across 36 views
        value_losses, not_ended_mask = self._normalize_loss_across_viewing_angles(batch_size, q_values_estimate, q_values_target, ref=ref)
        # average across the batch
        # scalar loss
        return torch.sum(value_losses) / torch.sum(not_ended_mask)

    def make_tr_instructions_by_t(self, tr_key_pairs, tr_timesteps):
        '''
        Extract language instructions from history buffer
        Arguments:
            tr_key_pairs: list of len(batch_size) tuples, each (instr_id, global_iter_idx).
            tr_timesteps: list of len(batch_size) integers, each indicate sampled timestep
        '''
        # Extract language instruction and convert to token IDs
        # self.env defined in subclass
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
        tensor_dtype = torch.long if var_name == 'action' else torch.float

        # Create container to hold training data
        variable_dim = self.history_buffer[tr_key_pairs[0]][var_name][tr_timesteps[0]].shape[0]
        # shape (batch_size, variable size)
        variable = torch.empty(len(tr_key_pairs), variable_dim, dtype=tensor_dtype, device=self.device)

        # Fill container with data extracted from history buffer
        for i, (key_pair, timestep) in enumerate(zip(tr_key_pairs, tr_timesteps)):
            variable[i,:] = self.history_buffer[key_pair][var_name][timestep]

        return variable

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
        actions = torch.empty(len(tr_key_pairs), action_dim, dtype=torch.long, device=self.device)

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
        mask = torch.zeros(len(tr_key_pairs), dtype=torch.bool, device=self.device)

        # Fill container with data extracted from history buffer
        for i, (key_pair, timestep) in enumerate(zip(tr_key_pairs, tr_timesteps)):
            mask[i] = int(view_ix in self.history_buffer[key_pair]['view_index_mask_indices'][timestep])

        return mask

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
            features[i,:] = self.env.env.features[long_id][view_ix, :]

        return torch.from_numpy(features).to(self.device)

    def get_rollout_view_indexed_features(self, obs, view_ix):
        '''
        Extract image features associated with a viewIndex(rotation), obs scan and viewpointIndex.

        Arguments:
            obs : a list of observations returned from env._get_obs()
            view_ix: integer [0-35] used to index panoramic images in the Matterport 3D simulator. 

        Returns: 
            (batch_size, feature size)        
        '''
        # Create container to hold training data
        feature_dim = obs[0]['feature'].shape[0]
        # shape (batch_size, feature size)
        features = np.empty((len(obs), feature_dim), dtype=np.float32) 

        # Fill container with data extracted from history buffer
        for i, ob in enumerate(obs):
            scan_id = ob['scan']
            viewpoint_idx = ob['viewpoint']
            long_id = scan_id + '_' + viewpoint_idx
            features[i,:] = self.env.env.features[long_id][view_ix, :]

        return torch.from_numpy(features).to(self.device)

    def _setup(self, env, explore_env, feedback):
        # Leave for subclasses to:
        # - setup nav, ask, recover feedback
        # - setup nav, ask, recover losses
        # - setup self.env
        # - add scans to teachers/advisors
        raise NotImplementedError('Subclasses are expected to implement _setup')

    def test(self, env, feedback, explore_env, use_dropout=False, allow_cheat=False):
        ''' Evaluate once on each instruction in the current environment '''

        self.is_eval = True
        # messy legacy setup in train.py()
        self.compute_rollout_loss = use_dropout

        self._setup(env, explore_env, feedback)
        if use_dropout:
            self.model.train()
        else:
            self.model.eval()
        return self.base_test()

    def train(self, env, optimizer, n_iters, feedback, idx, explore_env):
        '''
        Train for a given number `n_iters` of iterations. 
        `n_iters` is set to hparams.log_every (default 1000) or remaining.
        '''
        interval_training_iter_start_time = time.time()

        self.is_eval = False
        self.compute_rollout_loss = False

        # Set up self.env, feedback(teacher/argmax/learned)
        # Initialize losses, add scans to teachers/advisors
        self._setup(env, explore_env, feedback)

        # do not use dropout
        self.model.train()

        # Check expert roll-in prob
        assert self.beta >= 0.0 # always >= 0.0 in exponential decay

        # time report
        time_report = defaultdict(int)

        # List of 10*batch_size number of trajectory dictionaries
        last_traj = []

        # Loop through 1000 batches of datapts
        for itr in range(1, n_iters+1):

            # Global training iteration index
            global_iter_idx = idx + itr
            print ("Training, global_iter_idx = {}".format(global_iter_idx))

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
            if len(self.history_buffer) >= self.min_history_to_learn:
                
                # Sample a minibatch from the history buffer
                start_time = time.time()
                sampled_training_batch = self.history_buffer.sample_minibatch(self.tr_batch_size)
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
                optimizer.zero_grad()
                self.loss.backward()

                # Divide up gradients by 1/n_ensemble in the shared encoder if we are bootstrapping
                if self.bootstrap:
                    for param in self.model.decoder.parameters():
                        if param.grad is not None:
                            param.grad.data *= 1.0/float(self.n_ensemble)
                    for param in self.model.encoder.parameters():
                        if param.grad is not None:
                            param.grad.data *= 1.0/float(self.n_ensemble)

                # NOTE add code here if gradient clipping

                # Update torch model params
                optimizer.step()
                time_report['backprop_time'] += time.time() - start_time

            # Exponential decay expert-rollin probability beta
            if global_iter_idx >= self.start_beta_decay and global_iter_idx % self.decay_beta_every == 0:
                self.beta *= self.beta_decay_rate
                print('New expert roll-in probability %f' % self.beta)

        time_report['per_training_interval'] += time.time() - interval_training_iter_start_time
        return last_traj, time_report