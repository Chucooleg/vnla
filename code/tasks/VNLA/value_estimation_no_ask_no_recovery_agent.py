# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import division

import json
import os
import sys
import numpy as np
import random
import time
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributions as D
from torch import optim
import torch.nn.functional as F

from utils import padding_idx
from value_estimation_agent import ValueEstimationAgent
from oracle import make_oracle
from env import VNLAExplorationBatch

class ValueEstimationNoAskNoRecoveryAgent(ValueEstimationAgent):
    '''
    Value Estimation Agent with NO asking mechanism and NO recovery strategy.
    Algorithm 1 sketch with basic Aggrevate implementation
    '''

    # NOTE add recovery attributes here for recovery agents 

    def __init__(self, model, hparams, device):
        super(ValueEstimationNoAskNoRecoveryAgent, self).__init__(model, hparams, device)

        self.num_recent_frames = hparams.num_recent_frames

        # Used to initialize exploration environment, may not be necessary
        self.img_features = hparams.img_features
        self.agent_end_criteria = hparams.agent_end_criteria

    @staticmethod
    def n_input_nav_actions():
        # inherited this attribute from NavAgent
        return len(ValueEstimationNoAskNoRecoveryAgent.nav_actions)

    @staticmethod
    def n_output_nav_actions():
        # inherited this attribute from NavAgent
        # cannot output <start> or <ignore> so -2
        return len(ValueEstimationNoAskNoRecoveryAgent.nav_actions) - 2      

    @staticmethod
    def n_input_ask_actions():
        # inherited this attribute from ValueEstimationAgent
        return len(ValueEstimationNoAskNoRecoveryAgent.ask_actions)

    @staticmethod
    def n_output_ask_actions():
        # inherited this attribute from ValueEstimationAgent
        # cannot output <start> or <ignore> so -2
        return len(ValueEstimationNoAskNoRecoveryAgent.ask_actions) - 2 

    def _setup(self, env, explore_env, feedback):
        '''called in train() in parent class'''

        self.nav_feedback = feedback['nav']
        assert self.nav_feedback in self.feedback_options

        self.env = env
        self.frontier_explore_env = explore_env
        self.value_teacher.add_scans(env.scans)

        # track loss for training.
        self.losses = [] # to accumulate more all types of losses (nav, ask, value,...)
        self.loss_types = ['value_losses']
        self.value_losses = []
        self.uncertainties = []

    def _populate_agent_state_to_obs(self, obs, ended):
        """modify obs in-place (but env is not modified)
        for nav oracle to compute nav target 
        """
        for i, ob in enumerate(obs):
            ob['ended'] = ended[i]      

    def _compute_loss_training(self, global_iter_idx=None, traj_len=1.0):
        '''computed once at the end of every sampled mini-batch from history buffer'''

        # scalar - sum of one batch, all timesteps
        self.loss = self.value_loss
        # + self.ask_loss for ask agents
        # + self.recover_loss for recovery agents

        # Normalize loss by trajectory length
        iter_loss_avg = self.loss.item() / traj_len # get scalar from torch
        self.losses.append(iter_loss_avg)

        iter_value_loss_avg = self.value_loss.item() / traj_len
        self.value_losses.append(iter_value_loss_avg)

        if self.bootstrap:
            iter_uncertainty_avg = self.uncertainty.item() / traj_len
            self.uncertainties.append(iter_uncertainty_avg)

    def _compute_loss_rollout(self, global_iter_idx=None, traj_len=1.0):
        '''computed once at the end of every sampled mini-batch from history buffer'''

        # scalar - sum of one batch, all timesteps
        self.loss = self.rollout_value_loss
        # + self.ask_loss for ask agents
        # + self.recover_loss for recovery agents

        # Normalize loss by trajectory length
        iter_loss_avg = self.loss.item() / traj_len # get scalar from torch
        self.losses.append(iter_loss_avg)

        iter_value_loss_avg = self.rollout_value_loss.item() / traj_len
        self.value_losses.append(iter_value_loss_avg)

        if self.bootstrap:
            iter_uncertainty_avg = self.rollout_uncertainty.item() / traj_len
            self.uncertainties.append(iter_uncertainty_avg)

    def pred_training_batch(self, training_batch, global_iter_idx, output_res=False):
        '''
        Agent makes prediction on training batch sampled from history buffer.
        Compute self.loss, accumulate self.losses, self.value_losses
        training_obs :    returned from HistoryBuffer.sample_minibatch()
        '''
        # time keeping
        time_report = defaultdict(int)
        pred_start_time = time.time()
        start_time = time.time() 

        print("Training minibatch sampled from history buffer.")

        tr_key_pairs, tr_timesteps = training_batch
        batch_size = len(tr_timesteps)

        # Optional result book keeping
        training_batch_results = []
        if output_res:
            training_batch_results = [{
                # buffer indices
                'instr_id': instr_iter_key_pair[0],
                'global_iter_idx': instr_iter_key_pair[1],
            
                # for analysis
                'start_timestep': timestep,

                # q_values
                'teacher_values_target': [],
                'agent_values_estimate': [],

            } for (instr_iter_key_pair, timestep) \
                in zip(tr_key_pairs, tr_timesteps)]

        # Index initial command
        # One command for all time steps in No Ask Agents
        seq, seq_mask, seq_lengths = self.make_tr_instructions_by_t(tr_key_pairs, tr_timesteps)

        # Encode initial command
        ctx, _ = self.model.encode(seq, seq_lengths)

        # Initiate dummy Ask Actions
        # This tensor is always dont_ask for decoder in No Ask agents
        ques_out_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * \
            self.ask_actions.index('dont_ask')

        # time keeping
        time_report['initial_setup_training'] += time.time() - start_time

        # -----------------------------------------------------------------
        # LSTM unfold through the recent history frames e.g. t-3, t-2, t-1, t-0
        # Purpose: to compute decoder_h and cov from history
        start_time = time.time()

        # Initialize decoder hidden state
        decoder_h = None

        # Optional coverage vector
        if self.coverage_size is not None:
            # cov has shape  (batch size, max sequence len, coverage size)
            cov = torch.zeros(seq_mask.size(0), seq_mask.size(1),
                              self.coverage_size,
                              dtype=torch.float, device=self.device)
        else:
            cov = None   

        # Prevent indexing problem if agent just started in trajectory
        # make sure <0 indices becomes 0s
        minus_t = lambda t, d: t - d if t - d > 0 else 0

        for delta_t in reversed(range(self.num_recent_frames)):

            # delta_t loop through 3, 2, 1, 0
            # t_ix loop through t-3, t-2, t-1, t
            # length - batch_size
            t_ix = [minus_t(tr_t, delta_t) for tr_t in tr_timesteps]

            # Get macro-action decision at history time t
            # shape (batch_size, max_macro_action_seq_len)
            a_out_t = self.get_tr_variable_by_t(tr_key_pairs, t_ix, 'action')

            # Get image feature as a result of the macro-action decision at history time t
            f_out_t = self.get_tr_variable_by_t(tr_key_pairs, t_ix, 'feature')

            # Run decoder forward pass
            decoder_h, _, cov, _, _ = self.model.decode_nav(a_out_t, ques_out_t, f_out_t, decoder_h, ctx, seq_mask, view_index_mask=None, cov=cov, pred_val=False)

        time_report['decode_training_batch_history'] += time.time() - start_time

        # -----------------------------------------------------------------
        # LSTM unfold one further step at the frontier
        start_time = time.time()
        # Initialize tensor to store q-value estimates
        if self.bootstrap:
            # shape (36, batch_size, n_ensemble)
            q_values_tr_estimate_heads = torch.empty(self.num_viewIndex, batch_size, self.n_ensemble, dtype=torch.float, device=self.device)
        else:
            # shape (36, batch_size)
            q_values_tr_estimate = torch.empty(self.num_viewIndex, batch_size, dtype=torch.float, device=self.device)

        # Loop through 36 view indices in the pano sphere
        # run 100 in parallel instead of 36*100 in parallel
        for view_ix in range(self.num_viewIndex):  # [0-35]

            # Get proposed macro-action that result in current viewIndex(rotation)
            # tensor shape (batch_size, max_macro_action_seq_len)
            a_proposed = self.get_tr_view_indexed_action_by_t(tr_key_pairs, tr_timesteps, view_ix)

            # Get image feature at current viewIndex(rotation)
            # tensor shape (batch_size, feature_size)
            f_proposed = self.get_tr_view_indexed_features_by_t(tr_key_pairs, tr_timesteps, view_ix)

            # Get mask at that viewIndex(rotation)
            # i.e. not all rotation angles can connect to other vertices on graph
            # tensor shape (batch_size,)
            viewix_mask = self.get_tr_view_indexed_mask_by_t(tr_key_pairs, tr_timesteps, view_ix)

            # If implementing Ask Agent
            # ques_asked = ...

            # Run decoder forward pass
            if self.bootstrap:
                # shape (batch_size, self.n_ensemble)
                _, _, q_values_tr_estimate_heads[view_ix], _ = self.model.decode_nav( 
                    a_proposed, ques_out_t, f_proposed, decoder_h, ctx, seq_mask, 
                    view_index_mask=viewix_mask, cov=cov, pred_val=True)
            else:
                # shape (batch_size, )
                _, _, q_values_tr_estimate[view_ix], _ = self.model.decode_nav( 
                    a_proposed, ques_out_t, f_proposed, decoder_h, ctx, seq_mask, 
                    view_index_mask=viewix_mask, cov=cov, pred_val=True)

        if self.bootstrap:
            # tensor shape (batch_size, 36, n_ensemble)
            q_values_tr_estimate_heads = q_values_tr_estimate_heads.transpose(0, 1)
        else:
            # tensor shape  (batch_size, 36)
            q_values_tr_estimate = q_values_tr_estimate.t()

        time_report['decode_training_batch_frontier'] += time.time() - start_time
        # -----------------------------------------------------------------
        # Get q_values_target from history buffer
        # tensor shape (batch_size, self.num_viewIndex)
        q_values_target = self.get_tr_variable_by_t(tr_key_pairs, tr_timesteps, 'q_values_target')

        # -----------------------------------------------------------------
        if self.bootstrap:
            # Compute mean and variance among heads for every task and every view angle
            # shape (batch_size, 36)
            q_values_tr_estimate_variance, _ = torch.var_mean(q_values_tr_estimate_heads, dim=2)
            assert q_values_tr_estimate_variance.shape == (batch_size, self.num_viewIndex)

            # Estimate uncertainty
            # shape (batch_size,)
            q_values_tr_uncertainty = torch.empty(batch_size, dtype=torch.float, device=self.device)
            for i in range(batch_size):
                no_mask_idx = torch.nonzero(q_values_target[i] != 1e9).squeeze(-1)
                assert len(no_mask_idx.shape) == 1 and no_mask_idx.shape[0] <= self.num_viewIndex
                q_values_tr_uncertainty[i] = torch.mean(q_values_tr_estimate_variance[i][no_mask_idx])
        # -----------------------------------------------------------------
        # Compute q-value loss
        start_time = time.time()
        # Compute scalar loss
        if self.bootstrap:
            # tensor shape (batch_size, self.n_ensemble)
            bootstrap_masks = self.get_tr_bootstrap_masks_by_t(tr_key_pairs, tr_timesteps)
            self.value_loss = self.normalize_loss_with_mask( 
                batch_size, q_values_tr_estimate_heads, q_values_target, bootstrap_masks)
            self.uncertainty = self.normalize_uncertainty(batch_size, q_values_tr_uncertainty)
        else:
            self.value_loss = self.normalize_loss( 
                batch_size, q_values_tr_estimate, q_values_target, ref=False)
        time_report['compute_training_value_loss'] += time.time() - start_time

        # Save per batch loss to self.loss for backprop
        # Append to *_losses for logging
        start_time = time.time()
        self._compute_loss_training(global_iter_idx, traj_len=1.0)
        time_report['compute_loss_per_training_batch'] += time.time() - start_time

        # Optional save training batch results
        if output_res:
            start_time = time.time()
            teacher_values_target_list = q_values_target.data.tolist()
            if self.bootstrap:
                agent_tr_estimate_list = q_values_tr_estimate_heads.data.tolist()
            else:
                agent_tr_estimate_list = q_values_tr_estimate.data.tolist()
            for i in range(len(tr_timesteps)):
                training_batch_results[i]['teacher_values_target'] = teacher_values_target_list[i]
                training_batch_results[i]['agent_values_estimate'] = agent_tr_estimate_list[i]
            time_report['save_training_batch_results'] += time.time() - start_time

        print ("finished training from history buffer. self.value_loss = {}".format(self.value_loss))
        if self.bootstrap:
            print ("finished training from history buffer. self.uncertainty = {}".format(self.uncertainty))

        # total time report
        time_report['total_training_batch_time'] += time.time() - pred_start_time
        return training_batch_results, time_report

    def rollout(self, global_iter_idx=None):

        # time keeping
        time_report = defaultdict(int)
        rollout_start_time = time.time()
        initial_setup_time = time.time()
        
        # Training vs Validation settings
        if self.is_eval:
            # Evaluation
            use_hist_buffer = False
            # always agents rollin
            expert_rollin_bool = 0
            # whether to compute and report loss
            compute_rollout_loss = self.compute_rollout_loss
            # False will use oracle traj len (shorter cheat for loss computation) 
            use_expected_traj_len = not compute_rollout_loss
        else:
            # Training
            use_hist_buffer = True
            # Draw expert-rollin boolean
            expert_rollin_bool = np.random.binomial(1, self.beta)
            # whenever agent rollin, also report rollout loss (but no backprop from this)
            compute_rollout_loss = not expert_rollin_bool
            # Use oracle traj len to train
            use_expected_traj_len = False
        # report settings
        print("{} mode, train from history buffer = {}, {} roll-in, compute rollout loss = {}"\
            .format(
                ('eval' if self.is_eval else 'training'), 
                (use_hist_buffer), 
                ('expert' if expert_rollin_bool else 'agent'), 
                (compute_rollout_loss)))

        # Reset environment
        start_time = time.time()
        obs = self.env.reset(use_expected_traj_len)
        batch_size = len(obs)
        time_report['initial_setup_env_reset'] += time.time() - start_time

        # History buffer requires that the same instr_id doesn't appear more than once in a batch
        if use_hist_buffer:
            assert len(set([ob['instr_id'] for ob in obs])) == len([ob['instr_id'] for ob in obs])

        # Start roll-out book keeping
        # one trajectory per ob
        start_time = time.time()
        traj = [{
            # indexing
            'scan': ob['scan'],
            'instr_id': ob['instr_id'],

            # trajectory
            'agent_path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],

            # nav related
            'agent_nav': [], # from env action, list of lists of tuples 

            # q-value related
            'agent_q_values': [],  
            'teacher_q_values': [],  # list of per timestep vectors
            'teacher_cost_togo': [],  # list of per timestep vectors
            'teacher_cost_stepping': [],  # list of per timestep vectors

            # rollin
            'beta': None if self.is_eval else self.beta,
            'expert_rollin_bool': 0 if self.is_eval else expert_rollin_bool,

            # bootstrapping
            'agent_q_values_uncertainty': [],
            'agent_q_values_votes': [],
        } for ob in obs]
        time_report['initial_setup_initialize_traj_bookkeeping'] += time.time() - start_time

        # Index initial command
        start_time = time.time()
        # Stays the same in No Ask Agents
        seq, seq_mask, seq_lengths = self.make_instruction_batch(obs)
        time_report['initial_setup_make_instructions'] += time.time() - start_time

        # Encode initial command
        start_time = time.time()
        # ctx will not update in No Ask Agents
        # tensor shape (100, 8, 512)
        ctx, _ = self.model.encode(seq, seq_lengths)
        time_report['initial_setup_model_encode'] += time.time() - start_time

        # Initialize rotation actions
        start_time = time.time()
        # tensor shape (batch_size, max_macro_action_seq_len)
        a_t = torch.ones(batch_size, self.max_macro_action_seq_len, dtype=torch.long, device=self.device) * \
            self.nav_actions.index('<start>')
        time_report['initial_setup_initialize_actions'] += time.time() - start_time

        # Initialize dummy ask action
        start_time = time.time()
        # this tensor is always don't ask in No Ask Agents
        # shape (batch_size,)
        ques_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * \
            self.ask_actions.index('dont_ask')
        time_report['initial_setup_initialize_questions'] += time.time() - start_time

        # Get initial image features at starting position
        start_time = time.time()
        # shape (batch_size, feature size=2048)
        f_t = self.get_feature_variable(obs)
        time_report['initial_setup_get_features'] += time.time() - start_time

        # Initialize trajectory 'end' markers
        start_time = time.time()
        # Whether agent has deided to <stop>
        ended = np.array([False] * batch_size)
        ended_by_q_value = np.array([False] * batch_size)
        time_report['initial_setup_initialize_ends'] += time.time() - start_time

        # Initialize nav loss to be accumulated across batch
        if compute_rollout_loss:
            self.rollout_value_loss = 0
            self.rollout_value_ref_loss = 0
            if self.bootstrap:
                self.rollout_uncertainty = 0 

        # Initialize rotation actions for env (i.e. batch of simulators)
        # each element is a list of max_macro_action_seq_len tuples
        env_rotations = [None] * batch_size
        # Initialize step-forward actions for env (i.e. batch of simulators)
        # each element (1,0,0) or (0,0,0)
        env_stepping = [None] * batch_size

        # Initialize ^T time budget, different for each data point
        # Precomputed per trajectory when Env() was initialized
        episode_len = max(ob['traj_len'] for ob in obs)
        # print ("maximum episode length in batch = {}".format(episode_len))

        start_time = time.time()
        if use_hist_buffer:
            # Initialize experience batch at <start> when t=0
            # A new experience batch will be saved at every following time step
            experience_batch_t = [{
                # buffer indexing
                'instr_id': ob['instr_id'],
                'global_iter_idx': global_iter_idx,
                'viewpointIndex': ob['viewpoint'],
                'scan': ob['scan'],

                # training data - recent history
                # length = (instruction sentence length)
                'instruction': ob['instruction'],
                # tensors shape (image feature size)
                'feature': f_t[i],
                # tensor shape (self.max_macro_action_seq_len)
                'action': a_t[i],

                # training data - pano sphere
                # None at <start> pos when t=0
                'view_index_mask_indices': None,
                'viewix_actions_map': None,
                
                # training target
                'q_values_target': None,
            } for (i, ob) in enumerate(obs)]        

            # Add experience, remove earliest experiences if buffer is full
            self.history_buffer.add_experience(experience_batch_t)
            # print("Added new experience, history buffer size = {}".format(self.history_buffer.curr_buffer_size()))
        time_report['initial_setup_write_to_hist_buffer'] += time.time() - start_time

        # Initialize local buffer at <start>, append per timestep
        # Keep past j frames i.e.{} for continuous LSTM decoding
        start_time = time.time()
        local_buffer = [{
            'a_t': a_t,
            'f_t': f_t,
            # Below are skipped -- always the same in No Ask Agents
            # 'ques_t': ques_t,
            # 'ctx': ctx,
            # 'seq_mask': seq_mask,
        }]
        time_report['initial_setup_write_to_local_buffer'] += time.time() - start_time

        # time keeping
        time_report['initial_setup'] += time.time() - initial_setup_time

        # Loop through time budget ^T
        # 1 time step = 1 rotation + 1 step forward
        timestep = 1
        while timestep <= episode_len:

            # Sample head `K`
            active_head = np.random.randint(self.n_ensemble)

            # Modify obs in-place to indicate if any ob has 'ended'
            start_time = time.time()
            self._populate_agent_state_to_obs(obs, ended)
            time_report['pop_state_to_obs'] += time.time() - start_time

            # ---------------------------- Explore Frontier ----------------------------

            start_time = time.time()
            self.frontier_explore_env.reset_explorers(obs)
            time_delta = time.time() - start_time
            if timestep == 1:
                time_report['reset_frontier_explore_env_timestep_1'] += time_delta
            else:
                time_report['reset_frontier_explore_env_timestep_x'] += time_delta

            # Oracle provide instructions for the simulators to explore(turn) around
            # 3 lists of tuples, each of len batch_size.
            start_time = time.time()
            explore_instructions = self.value_teacher.make_explore_instructions(obs)
            time_report['make_explore_instructions'] += time.time() - start_time

            # Simulators explore around, map panoramic view index to actions
            # and reachable viewpoint Indices
            start_time = time.time()
            # list (batch_size,). each viewIndex int
            curr_viewIndex = [ob['viewIndex'] for ob in obs]
            # list (batch_size, 36, varies). Each [(0,1,0), (0,-1,0), (0,0,1)...]
            # list (batch_size, 36). each viewpointId string
            # arr shape(36, batch_size), each mask boolean.
            viewix_env_actions_map, viewix_next_vertex_map, view_index_mask, explore_time_report = \
                self.frontier_explore_env.explore_sphere(explore_instructions, curr_viewIndex, self.num_viewIndex, timestep)
            # Make sure that all env action sequences have valid length
            for i in range(len(viewix_env_actions_map)):
                for j in range(len(viewix_env_actions_map[i])):
                    assert len(viewix_env_actions_map[i][j]) <= self.max_macro_action_seq_len
            time_report['explore_sphere'] += time.time() - start_time

            # Keep time for the sphere exploration processes
            for key in explore_time_report.keys():
                time_report[key] += explore_time_report[key]

            # Oracle compute q-value targets
            start_time = time.time()
            # arr shape (batch_size, self.num_viewIndex=36)
            # arr shape (batch_size, )
            # arr shape (batch_size, self.num_viewIndex=36)
            # arr shape (batch_size, self.num_viewIndex=36)
            q_values_target_arr, end_target, cost_togo_arr, cost_stepping_arr = self.value_teacher.compute_frontier_costs(obs, viewix_next_vertex_map, timestep)
            # tensor shape (batch_size, self.num_viewIndex=36)
            q_values_target = torch.tensor(q_values_target_arr, dtype=torch.float, device=self.device)

            time_report['make_q_targets'] += time.time() - start_time
            
            # Oracle map env level actions back to action indices as LSTM input
            # actions are either rotation or <ignore>, no <end> or forward
            # tensor shape(36, batch_size, self.max_macro_action_seq_len)
            start_time = time.time()
            viewix_actions_map = self.value_teacher.translate_env_actions(
                obs, viewix_env_actions_map, self.max_macro_action_seq_len, self.num_viewIndex)
            viewix_actions_map = torch.from_numpy(viewix_actions_map).long().to(self.device)
            time_report['make_viewix_actions_map'] += time.time() - start_time

            # ---------------------------- ----------------------------
            # Determine next macro action sequence by expert or agent
            start_time = time.time()
            # tensor shape (36, 100) if agent rollin
            q_values_rollout_estimate = None
            q_values_rollout_uncertainty = None
            # arr (batch_size) if agent rollin
            end_estimated = None

            if expert_rollin_bool:

                expert_select_start_time = time.time()

                # Select the best view direction determined by expert
                # tensor shape (batch_size,)
                best_view_ix = self.argmax(-q_values_target)
                # tensor shape (1, batch_size, self.max_macro_action_seq_len)
                best_view_ix_tiled = best_view_ix.view(-1, 1).repeat(1, viewix_actions_map.shape[2]).unsqueeze(0)
                assert best_view_ix_tiled.shape == (1, batch_size, self.max_macro_action_seq_len)

                # Select the macro-rotation associated with the best view direction
                # only rotation or ignore
                # tensor shape (batch_size, self.max_macro_action_seq_len)
                a_t =  viewix_actions_map.gather(0, best_view_ix_tiled).squeeze()
                assert a_t.shape == (batch_size, self.max_macro_action_seq_len)

                time_report['select_expert_macro_action'] += time.time() - expert_select_start_time

            else:

                # -------------- LSTM unfold through recent history (j frames)
                # Purpose: only to get the last decoder_h and cov
                decode_hist_start_time = time.time()

                # Initialize decoder hidden state
                decoder_h = None

                # Optional coverage vector
                if self.coverage_size is not None:
                    # cov has shape  (batch size, max sequence len, coverage size)
                    cov = torch.zeros(seq_mask.size(0), seq_mask.size(1), self.coverage_size,
                                    dtype=torch.float, device=self.device)
                else:
                    cov = None                 

                for t in range(len(local_buffer)):
                    # t loop through 0, 1, 2, 3, ... j

                    # tensor shape (batch_size, self.max_macro_action_seq_len)
                    a_out_t = local_buffer[t]['a_t']
                    # tensor shape (batch_size, feature size)
                    f_out_t = local_buffer[t]['f_t']

                    decoder_h, _, cov, _, _ = self.model.decode_nav(a_out_t, ques_t, f_out_t, decoder_h, ctx, seq_mask, view_index_mask=None, cov=cov, pred_val=False)

                time_report['decode_history'] += time.time() - decode_hist_start_time

                # -------------- LSTM unfold one step further at pano sphere
                decode_frontier_start_time = time.time()

                # Initialize tensor to store q-value estimates
                if self.bootstrap:
                    # tensor shape (36, batch_size, self.n_ensemble)
                    q_values_rollout_estimate_heads = torch.empty( 
                        self.num_viewIndex, batch_size, self.n_ensemble, 
                        dtype=torch.float, device=self.device)
                else:
                    # tensor shape (36, 100)
                    q_values_rollout_estimate = torch.empty( 
                        self.num_viewIndex, batch_size, dtype=torch.float, device=self.device)

                # Loop through 36 view indices in the pano sphere
                # run 100 in parallel instead of 36*100 in parallel
                for view_ix in range(self.num_viewIndex):  # i.e. [0-35]
                    # Run per batch LSTM computations below

                    # Get proposed macro-action that result in current viewIndex(rotation)
                    # tensor shape (batch_size, self.max_macro_action_seq_len)
                    get_a_proposed_start_time = time.time()
                    a_proposed = viewix_actions_map[view_ix]
                    time_report['decode_frontier_get_action'] += time.time() - get_a_proposed_start_time

                    # Get image feature at current viewIndex(rotation)
                    # tensor shape (batch_size, feature_size)
                    get_f_proposed_start_time = time.time()
                    f_proposed = self.get_rollout_view_indexed_features(obs, view_ix)
                    time_report['decode_frontier_get_feature'] += time.time() - get_f_proposed_start_time
                    
                    # Get mask at that viewIndex(rotation)
                    # i.e. not all rotation angles can connect to other vertices on graph
                    # tensor shape (batch_size,)
                    get_viewix_mask_start_time = time.time()
                    view_ix_mask = torch.tensor(view_index_mask[view_ix], dtype=torch.bool, device=self.device)
                    time_report['decode_frontier_get_viewix_mask'] += time.time() - get_viewix_mask_start_time

                    # If implementing Ask Agent
                    # ques_asked = ...

                    # Run decoder forward pass
                    frontier_decode_time = time.time()
                    if self.bootstrap:
                        # shape (batch_size, self.n_ensemble)
                        _, _, q_values_rollout_estimate_heads[view_ix], _ = self.model.decode_nav( 
                            a_proposed, ques_t, f_proposed, decoder_h, ctx, seq_mask,
                            view_index_mask=view_ix_mask, cov=cov, pred_val=True)
                    else:
                        # shape (batch_size,)
                        _, _, q_values_rollout_estimate[view_ix], _ = self.model.decode_nav( 
                            a_proposed, ques_t, f_proposed, decoder_h, ctx, seq_mask,
                            view_index_mask=view_ix_mask, cov=cov, pred_val=True)

                    time_report['decode_frontier_decoder_forward'] += time.time() - frontier_decode_time

                time_report['decode_frontier'] += time.time() - decode_frontier_start_time

                # -------------- Compute Uncertainty
                compute_uncertainty_start_time = time.time()

                # Compute Variance, Mean/Sample q-value among heads (if bootstrap)
                if self.bootstrap:
                    # shape (batch_size, 36, n_ensemble)
                    q_values_rollout_estimate_heads = q_values_rollout_estimate_heads.transpose(0, 1)

                    # Training mode
                    # Compute variance across heads
                    # tensor shape (batch_size, 36)
                    q_values_rollout_estimate_variance = torch.var(q_values_rollout_estimate_heads, dim=2)
                    assert q_values_rollout_estimate_variance.shape == (batch_size, self.num_viewIndex)

                    # Estimate uncertainty across viewing angles
                    # shape (batch_size,)
                    q_values_rollout_uncertainty = torch.empty(batch_size, dtype=torch.float, device=self.device)
                    for i in range(batch_size):
                        no_mask_idx = torch.nonzero(q_values_target[i] != 1e9).squeeze(-1)
                        assert len(no_mask_idx.shape) == 1 and no_mask_idx.shape[0] <= self.num_viewIndex
                        q_values_rollout_uncertainty[i] = torch.mean(q_values_rollout_estimate_variance[i][no_mask_idx])

                time_report['select_agent_macro_action'] += time.time() - compute_uncertainty_start_time
                # -------------- Select Next Action
                agent_select_start_time = time.time()

                # Voting for next action -- if bootstrap eval
                if self.bootstrap:
                    if self.is_eval:
                        # Each head vote for their argmin viewing angle
                        # tensor shape (batch_size, self.n_ensemble)
                        votes = torch.stack([self.argmax(-q_values_rollout_estimate_heads[:, :, h]) for h in range(self.n_ensemble)], dim=1)
                        assert votes.shape == (batch_size, self.n_ensemble)

                        # Find majority vote and a matching head
                        # voted viewing angle
                        # tensor shape (batch_size, )
                        best_view_ix, _ = torch.mode(votes)
                        assert best_view_ix.shape == (batch_size, )

                        # Average q_value estimates from matching heads
                        # tensor shape (batch_size, 1, self.n_ensemble)
                        matching_boolean = (votes == best_view_ix.view(-1, 1)).unsqueeze(1).float()
                        assert matching_boolean.shape == (batch_size, 1, self.n_ensemble)
                        # tensor shape (batch_size, self.num_viewIndex) / tensor shape (batch_size, 1)
                        # = tensor shape (batch_size, self.num_viewIndex)
                        q_values_rollout_estimate = torch.sum(q_values_rollout_estimate_heads * matching_boolean, dim=2) / torch.sum(matching_boolean, dim=2)
                        assert q_values_rollout_estimate.shape == (batch_size, self.num_viewIndex)

                        # Select q-values that matches the majority voted viewing angle
                        # best_view_h_tiled = best_view_h.view(-1, 1).repeat(1, self.num_viewIndex).unsqueeze(-1)
                        # assert best_view_h_tiled.shape == (batch_size, self.num_viewIndex, 1)
                        # q_values_rollout_estimate = q_values_rollout_estimate_heads.gather(2, best_view_h_tiled).squeeze(-1)
                        # assert q_values_rollout_estimate.shape == (batch_size, self.num_viewIndex)

                    else:
                        # Follow head k during training
                        # tenosr shape (batch_size, self.num_viewIndex=36)
                        q_values_rollout_estimate = q_values_rollout_estimate_heads[:, :, active_head]
                        assert q_values_rollout_estimate.shape == (batch_size, self.num_viewIndex)

                        # Select the best view direction determined by expert
                        # tensor shape (batch_size, ), gradient detached
                        best_view_ix = self.argmax(-q_values_rollout_estimate)
                        assert best_view_ix.shape == (batch_size, )
                        
                else:
                    # Reshape q-value estimates tensor back to q_val    
                    # shape (batch_size, self.num_viewIndex=36)
                    # .t() expects 2-D tensor
                    q_values_rollout_estimate = q_values_rollout_estimate.t()
                    assert q_values_rollout_estimate.shape == (batch_size, self.num_viewIndex)

                    # Select the best view direction determined by expert
                    # tensor shape (batch_size, ), gradient detached
                    best_view_ix = self.argmax(-q_values_rollout_estimate)
                    assert best_view_ix.shape == (batch_size, )

                # Select the macro-rotation associated with the best view direction
                # tensor shape (1, batch_size, self.max_macro_action_seq_len)
                best_view_ix_tiled = best_view_ix.view(-1, 1).repeat(1, viewix_actions_map.shape[2]).unsqueeze(0)
                assert best_view_ix_tiled.shape == (1, batch_size, self.max_macro_action_seq_len)
                # tensor shape (batch_size, max_macro_action_seq_len), only rotation or <ignore>
                a_t = viewix_actions_map.gather(0, best_view_ix_tiled).squeeze()
                assert a_t.shape == (batch_size, self.max_macro_action_seq_len)

                # Compute end markers
                # (end after upcoming rotation and step-forward)
                # array shape (batch_size,)
                end_estimated = (torch.min(q_values_rollout_estimate, dim=1)[0] <= self.agent_end_criteria).cpu().data.numpy()
                assert end_estimated.shape[0] == batch_size

                time_report['select_agent_macro_action'] += time.time() - agent_select_start_time
                # -------

            time_report['compute_next_nav_by_feedback'] += time.time() - start_time
            # ----------------------------

            # Compute rollout loss to track validation performance
            if compute_rollout_loss:
                start_time = time.time()
                # scalar
                self.rollout_value_loss += self.normalize_loss(batch_size, q_values_rollout_estimate, q_values_target, ref=False)
                self.rollout_value_ref_loss += self.normalize_loss(batch_size, q_values_rollout_estimate, q_values_target, ref=True)
                if self.bootstrap:
                    self.rollout_uncertainty += self.normalize_uncertainty(batch_size, q_values_rollout_uncertainty)
                # torch.mean(q_values_rollout_uncertainty)
                time_report['compute_rollout_value_loss_with_critierion'] += time.time() - start_time

            # Translate chosen macro-rotations to env actions
            start_time = time.time()
            # only rotation or ignore
            a_t_list = a_t.data.tolist()
            for i, ob in enumerate(obs):
                # list of length agent.max_macro_action_seq_len
                # e.g. [(0, 1, 0), (0, 1, -1), ..... (0,0,0)]
                env_rotations[i] = self.value_teacher.interpret_agent_rotations(a_t_list[i], ob)
                # forward (1,0,0) or ignore (0,0,0)
                env_stepping[i] = self.value_teacher.interpret_agent_forward(ob)
            time_report['translate_to_env_actions'] += time.time() - start_time

            # Simulators (batch) take the chosen env rotations
            start_time = time.time()
            # a sim can rotate multiple times
            obs = self.env.steps(env_rotations)
            time_report['env_rotation_step'] += time.time() - start_time

            # Get image features after rotation
            # tensor (batch_size, feature size)
            start_time = time.time()
            f_t = self.get_feature_variable(obs)
            time_report['get_feature_after rotation'] += time.time() - start_time

            # Convert to batch_size first for experience update
            # tensor shape (batch_size, 36, self.max_macro_action_seq_len)
            start_time = time.time()
            viewix_actions_map = viewix_actions_map.transpose(0, 1)
            # array shape (batch_size, 36)
            view_index_mask = view_index_mask.transpose()

            # Convert tensor and arrays to list for saving to json
            # logging only
            if not expert_rollin_bool:
                q_values_rollout_estimate_list = q_values_rollout_estimate.data.tolist()
                if self.bootstrap:
                    q_values_rollout_uncertainty_list = q_values_rollout_uncertainty.data.tolist()
                    if self.is_eval:
                        votes_list = votes.data.tolist()
            time_report['prepare_tensors_for_saving'] += time.time() - start_time

            # Initialize new experience_batch_t
            # length <= batch_size because some ob may have ended already
            start_time = time.time()
            experience_batch_t = []

            # Convert np array to list for traj logging
            q_values_target_list = q_values_target_arr.tolist()
            cost_togo_list = cost_togo_arr.tolist()
            cost_stepping_list = cost_stepping_arr.tolist()
            # Update experience batch & traj post rotation
            for i, ob in enumerate(obs):
                if not ended[i]:
                    if use_hist_buffer:
                        experience_batch_t.append({
                            # buffer indexing
                            'instr_id': ob['instr_id'],
                            'global_iter_idx': global_iter_idx,
                            'viewpointIndex': ob['viewpoint'],
                            'scan': ob['scan'],

                            # training data - recent history
                            # length = (instruction sentence length)
                            'instruction': ob['instruction'],
                            # tensors shape (image feature size)
                            'feature': f_t[i],
                            # tensor shape (self.max_macro_action_seq_len)
                            # rotation only. (won't save <ignore> because ended traj wont be added)
                            'action': a_t[i],

                            # training data - pano sphere
                            # np array shape (varies)
                            'view_index_mask_indices': np.nonzero(view_index_mask[i])[0],
                            # tensors shape (self.num_viewIndex=36, self.max_macro_action_seq_len)
                            'viewix_actions_map': viewix_actions_map[i],

                            # training target
                            # tensors shape (self.num_viewIndex=36)
                            'q_values_target': q_values_target[i],
                        })
                    
                    # trajectory
                    traj[i]['agent_path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                    # add rotation action to all env actions
                    traj[i]['agent_nav'].append(env_rotations[i])
                    # q-values
                    traj[i]['agent_q_values'].append(None if expert_rollin_bool else q_values_rollout_estimate_list[i])
                    traj[i]['teacher_q_values'].append(q_values_target_list[i])
                    traj[i]['teacher_cost_togo'].append(cost_togo_list[i])
                    traj[i]['teacher_cost_stepping'].append(cost_stepping_list[i])

                    # bootstrapping
                    if self.bootstrap:
                        traj[i]['agent_q_values_uncertainty'].append(None if expert_rollin_bool else q_values_rollout_uncertainty_list[i])
                        traj[i]['agent_q_values_votes'].append(None if (expert_rollin_bool or not self.is_eval) else votes_list[i])
            time_report['write_experience_batch'] += time.time() - start_time

            if use_hist_buffer:
                # Write experience to history buffer
                start_time = time.time()
                self.history_buffer.add_experience(experience_batch_t)
                # print("Added new experience, history buffer size = {}".format(self.history_buffer.curr_buffer_size()))
                time_report['save_to_history_buffer'] += time.time() - start_time

            # Append tensors for the whole batch for this time step
            start_time = time.time()
            # Remove earliest form buffer if full
            if len(local_buffer) >= self.num_recent_frames:
                local_buffer.pop(0)
            local_buffer.append({
                'a_t': a_t,     #a out
                'f_t': f_t,     #f out
                # ask, instruction update in Ask Agents
            })
            time_report['save_to_local_buffer'] += time.time() - start_time

            # Simulators step forward to the direct-facing vertex after rotation
            start_time = time.time()
            obs = self.env.step(env_stepping)
            time_report['env_forward_step'] += time.time() - start_time

            # Update traj after stepping forward
            start_time = time.time()
            for i, ob in enumerate(obs):
                if not ended[i]:
                    # trajectory
                    traj[i]['agent_path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                    # add step-forward action to all env actions
                    traj[i]['agent_nav'].append(env_stepping[i])

                    # Mark if trajectory has ended
                    if expert_rollin_bool:
                        if end_target[i]:
                            ended_by_q_value[i] = True
                        if end_target[i] or timestep >= ob['traj_len']:
                            ended[i] = True
                    else:
                        if end_estimated[i]:
                            ended_by_q_value[i] = True
                        if end_estimated[i] or timestep >= ob['traj_len']:
                            ended[i] = True

            # Early exit if all trajectories in the batch has <end>ed
            if ended.all():
                print ("Ending all trajectories, last time step = {}".format(timestep))
                break
            time_report['save_traj_output'] += time.time() - start_time

            # Increment timestep for the next While def 
            timestep += 1

        # Compute rollout loss to track validation performance
        if compute_rollout_loss:
            start_time = time.time()
            print ("{} rollout value {} loss = {}".format(
                'eval' if self.is_eval else 'training', self.loss_function_str,
                self.rollout_value_loss/timestep))
            print ("{} rollout reference value {} loss = {}".format(
                'eval' if self.is_eval else 'training', self.loss_function_ref_str,
                self.rollout_value_ref_loss/timestep))
            if self.bootstrap:
                print ("{} rollout uncertainty = {}".format(
                    'eval' if self.is_eval else 'training', self.rollout_uncertainty/timestep))
            # if eval under training conditions to report validation loss
            if self.is_eval:
                self._compute_loss_rollout(global_iter_idx, traj_len=episode_len*1.0)
            time_report['report_rollout_value_loss'] += time.time() - start_time

        # Track how many trajectories are ended by applying q-value threshold
        print ("{} ended {} trajectories in batch.".format('expert' if expert_rollin_bool else 'agent', np.sum(ended_by_q_value)))

        time_report['total_rollout_time'] += time.time() - rollout_start_time
        return traj, time_report
