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

    def _setup(self, env, feedback):
        '''called in train() in parent class'''

        self.nav_feedback = feedback['nav']
        assert self.nav_feedback in self.feedback_options

        self.env = env
        self.value_teacher.add_scans(env.scans)

        self.losses = [] # to accumulate more all types of losses (nav, ask, value,...)
        self.loss_types = ['value_losses']
        self.value_losses = []    
        self.end_losses = []  

    def _populate_agent_state_to_obs(self, obs, ended):
        """modify obs in-place (but env is not modified)
        for nav oracle to compute nav target 
        """
        for i, ob in enumerate(obs):
            ob['ended'] = ended[i]      

    def _compute_loss(self, global_iter_idx=None):
        '''computed once at the end of every sampled mini-batch from history buffer'''

        self.loss = self.value_loss + self.end_loss
        # + self.ask_loss for ask agents
        # + self.recover_loss for recovery agents

        loss = self.loss.item() # get scalar from torch
        self.losses.append(loss)

        value_loss = self.value_loss.item()
        self.value_losses.append(value_loss)

        end_loss = self.end_loss.item()
        self.end_losses.append(end_loss)        

        # training mode
        if global_iter_idx: 
            self.SW.add_scalar('train loss per iter', loss, global_iter_idx)
            self.SW.add_scalar('train value loss per iter', value_loss, global_iter_idx)
            self.SW.add_scalar('train end loss per iter', end_loss, global_iter_idx)

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

        tr_key_pairs, tr_timesteps = training_batch
        batch_size = len(tr_timesteps)

        # Optional result book keeping
        training_batch_results = {}
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
        # This tensor is always zeros for decoder in No Ask agents
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
            decoder_h, _, _, _, _, cov = self.model.decode_nav(a_out_t, ques_out_t, f_out_t, decoder_h, ctx, seq_mask, view_index_mask=None, cov=cov)

        time_report['decode_training_batch_history'] += time.time() - start_time

        # -----------------------------------------------------------------
        # LSTM unfold one further step at the frontier
        start_time = time.time()
        # Initialize tensor to store q-value estimates
        # shape (36, batch_size)
        # TODO require grad is True here?
        q_values_tr_estimate = torch.empty(self.num_viewIndex, batch_size, dtype=torch.float, device=self.device, requires_grad=True)
        end_tr_estimate = torch.empty(self.num_viewIndex, batch_size, dtype=torch.float, device=self.device, requires_grad=True)

        # Loop through 36 view indices in the pano sphere
        # run 100 in parallel instead of 36*100 in parallel
        for view_ix in range(self.num_viewIndex):  # [0-35]

            # Get proposed macro-action that result in current viewIndex(rotation)
            # tensor shape (batch_size, max_macro_action_seq_len)
            a_proposed = get_tr_view_indexed_action_by_t(tr_key_pairs, tr_timesteps, view_ix)

            # Get image feature at current viewIndex(rotation)
            # tensor shape (batch_size, feature_size)
            f_proposed = get_tr_view_indexed_features_by_t(tr_key_pairs, tr_timesteps, view_ix)

            # Get mask at that viewIndex(rotation)
            # i.e. not all rotation angles can connect to other vertices on graph
            # tensor shape (batch_size,)
            view_ix_mask = get_tr_view_indexed_mask_by_t(tr_key_pairs, tr_timesteps, view_ix)

            # Use existing decoder_h, cov computed from recent history
            # TODO what do we do with the gradient? detach()?

            # If implementing Ask Agent
            # ques_asked = ...

            # Run decoder forward pass
            decoder_h_curr, _, q_values_tr_estimate[view_ix], end_tr_estimate[view_ix], end_sigmoid, cov_curr = self.model.decode_nav(a_proposed, ques_out_t, f_proposed, decoder_h, ctx, seq_mask, view_index_mask=view_ix_mask, cov=cov)

        # Reshape q_values_tr_estimate for loss computation
        # to (batch_size, self.num_viewIndex)
        q_values_tr_estimate = q_values_tr_estimate.t()
        end_tr_estimate = end_tr_estimate.t()

        time_report['decode_training_batch_frontier'] += time.time() - start_time

        # -----------------------------------------------------------------
        # Compute q-value loss
        start_time = time.time()
        # Get q_values_target from history buffer
        # tensor shape (batch_size, self.num_viewIndex)
        q_values_target = self.get_tr_variable_by_t(tr_key_pairs, tr_timesteps, 'q_values_target')
        # tensor shape (batch_size, self.num_viewIndex)
        end_target = self.get_tr_end_target_by_t(tr_key_pairs, tr_timesteps)
        # Compute scalar loss
        self.value_loss = self.value_criterion(q_values_tr_estimate, q_values_target)
        self.end_loss = self.end_criterion(end_tr_estimate, end_target) # TODO implement critierion
        time_report['compute_value_loss'] += time.time() - start_time

        # Save per batch loss to self.loss for backprop
        # Append to *_losses for logging
        start_time = time.time()
        self._compute_loss(global_iter_idx)
        time_report['compute_loss_per_training_batch'] += time.time() - start_time

        # Optional save training batch results
        if output_res:
            start_time = time.time()
            for i in range(len(tr_timesteps)):
                training_batch_results[i]['teacher_values_target'] = q_values_target[i].tolist()
                training_batch_results[i]['agent_values_estimate'] = q_values_tr_estimate[i].tolist()
            time_report['save_training_batch_results'] += time.time() - start_time

        # total time report
        time_report['total_training_batch_time'] += time.time() - pred_start_time
        return training_batch_results, time_report

    def rollout(self, global_iter_idx=None):
        
        # time keeping
        time_report = defaultdict(int)
        rollout_start_time = time.time()
        start_time = time.time() 

        # Draw expert-rollin boolean
        if self.is_eval:
            expert_rollin_bool = 0
        else:
            expert_rollin_bool = np.random.binomial(1, self.beta)

        # Reset environment
        obs = self.env.reset(self.is_eval)
        batch_size = len(obs)     

        # Start roll-out book keeping
        # one trajectory per ob
        traj = [{
            # indexing
            'scan': ob['scan'],
            'instr_id': ob['instr_id'],

            # trajectory
            'agent_path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],

            # nav related
            'agent_nav': [], # from env action, list of lists of tuples 

            # q-value related
            'agent_q_values': [],  # list of per timestep vectors, if agent roll-in TODO, append empty if not agent roll-in
            'teacher_q_values': [],  # list of per timestep vectors

            # rollin
            'beta': None if self.is_eval else self.beta,
            'expert_rollin_bool': 0 if self.is_eval else expert_rollin_bool,
        } for ob in obs]

        # Index initial command
        # Stays the same in No Ask Agents
        seq, seq_mask, seq_lengths = self.make_instruction_batch(obs)

        # Encode initial command
        # ctx will not update in No Ask Agents
        ctx, _ = self.model.encode(seq, seq_lengths)

        # Initialize rotation actions
        # tensor shape (batch_size, max_macro_action_seq_len)
        a_t = torch.ones(batch_size, self.max_macro_action_seq_len, dtype=torch.long, device=self.device) * \
            self.nav_actions.index('<start>')

        # Initialize dummy ask action
        # this tensor is always zeros in No Ask Agents
        # shape (batch_size,)
        ques_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * \
            self.ask_actions.index('dont_ask')

        # Get initial image features at starting position
        # shape (batch_size, feature size)
        f_t = self.get_feature_variable(obs)

        # Initialize trajectory 'end' markers
        # Whether agent has deided to <stop>
        ended = np.array([False] * batch_size)  

        # Initialize rotation actions for env (i.e. batch of simulators)
        # each element is a list of max_macro_action_seq_len tuples
        env_rotations = [None] * batch_size
        # Initialize step-forward actions for env (i.e. batch of simulators)
        # each element (1,0,0) or (0,0,0)
        env_stepping = [None] * batch_size

        # Initialize ^T time budget, different for each data point
        # Precomputed per trajectory when Env() was initialized
        episode_len = max(ob['traj_len'] for ob in obs)  

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
            'end_target_indices': None,
            
            # training target
            'q_values_target': None,
        } for (i, ob) in enumerate(obs)]        

        # Remove earliest iteration of experience from buffer if buffer is full
        if self.history_buffer.is_full():
            self.history_buffer.remove_earliest_experience()  

        # Write <start> t=0 experience to history buffer
        self.history_buffer.add_experience(experience_batch_t)

        # Initialize local buffer at <start>, append per timestep
        # Keep past j frames i.e.{} for continuous LSTM decoding
        local_buffer = [{
            'a_t': a_t,
            'f_t': f_t,
            # Below are skipped -- always the same in No Ask Agents
            # 'ques_t': ques_t,  
            # 'ctx': ctx, 
            # 'seq_mask': seq_mask, 
        }]

        # time keeping
        time_report['initial_setup'] += time.time() - start_time

        # Loop through time budget ^T
        # 1 time step = 1 rotation + 1 step forward
        timestep = 1
        while timestep <= episode_len:

            # Modify obs in-place to indicate if any ob has 'ended'
            start_time = time.time()
            self._populate_agent_state_to_obs(obs, ended)
            time_report['pop_state_to_obs'] += time.time() - start_time


            # ---------------------------- ----------------------------

            # Explore pano sphere TODO

                # ARRAY
                # build view_index_mask, shape(36, batch_size), np array!
                view_index_mask = # arr shape(36, batch_size)

                end_target = # arr shape(36, batch_size)

                # TENSOR
                # build viewix_actions_map
                # list (batch_size, 36, varies)
                viewix_env_actions_map = 

                # don't change this shape
                viewix_actions_map = # arr shape(36, batch_size, self.max_macro_action_seq_len), tensor!

                # compute q_values_target tensor # shape (batch_size, self.num_viewIndex=36), tensor!
                # TODO remember to convert target to tensor for training in experience batch
                # TENSOR
                q_values_target = torch.tensor(q_values_target_list, dtype=torch.long, device=self.device)

            # ------- Try something here

            # Outsource panorama exploration to a dedicated batch of simulators 
            self.frontier_explore_env.reset(obs)

            # Oracle provide instructions for the simulators to explore(turn) around
            # 3 lists, each of len batch_size.
            explore_instructions = self.value_teacher.make_explore_instructions(obs)

            # Simulators explore around, map panoramix index to actions and reachable vertices
            # list (batch_size, 36, varies). Each [(0,1,0), (0,-1,1)]
            # list (batch_size, 36, varies). Each string.
            viewix_env_actions_map, viewix_next_vertex_map = \
                self.frontier_explore_env.explore_sphere(explore_instructions)

            # oracle
            # take viewix_next_vertex_map, make q_values_target, end_target
            # take viewix_env_actions_map, make viewix_actions_map





            # ---------------------------- ----------------------------
            # Determine next macro action sequence by expert or agent
            start_time = time.time()
            q_values_rollout_estimate = None

            if expert_rollin_bool:

                expert_select_start_time = time.time()

                # Select the best view direction determined by expert 
                # tensor shape (batch_size,)
                best_view_ix = self.argmax(-q_values_target)
                # tensor shape (1, batch_size, self.max_macro_action_seq_len)
                best_view_ix_tiled = best_view_ix.view(-1, 1).repeat(1, viewix_actions_map.shape[2]).unsqueeze(0)
                assert best_view_ix_tiled.shape == (1, batch_size, self.max_macro_action_seq_len)

                # Select the macro-action associated with the best view direction
                # tensor shape (batch_size, self.max_macro_action_seq_len)
                a_t =  viewix_actions_map.gather(0, best_view_ix_tiled).squeeze()
                assert a_t.shape == (batch_size, self.max_macro_action_seq_len)

                time_report['select_expert_macro_action'] += time.time() - expert_select_start_time

            else:
                # Sanity check local buffer size
                assert local_buffer and len(local_buffer) <= self.num_recent_frames

                # ------- LSTM unfold through recent history (j frames)
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

                    decoder_h, _, _, _, _, cov = self.model.decode_nav(a_out_t, ques_t, f_out_t, decoder_h, ctx, seq_mask, view_index_mask=None, cov=cov)
                time_report['decode_history'] += time.time() - decode_hist_start_time

                # ------- LSTM unfold one step further at pano sphere
                decode_frontier_start_time = time.time()

                # Initialize tensor to store q-value estimates
                # shape (36, 100)
                q_values_rollout_estimate = torch.empty(self.num_viewIndex, batch_size, dtype=torch.float, device=self.device)
                ended_rollout_estimate = torch.empty(self.num_viewIndex, batch_size, dtype=torch.float, device=self.device)

                # Loop through 36 view indices in the pano sphere
                # run 100 in parallel instead of 36*100 in parallel
                for view_ix in range(self.num_viewIndex):  # i.e. [0-35]
                    # Run per batch LSTM computations below

                    # Get proposed macro-action that result in current viewIndex(rotation)
                    # tensor shape (batch_size, self.max_macro_action_seq_len)
                    a_proposed = viewix_actions_map[view_ix]  

                    # Get image feature at current viewIndex(rotation)
                    # tensor shape (batch_size, feature_size)
                    f_proposed = torch.stack([self.env.features[ob['scan'] + '_' + ob['viewpoint']][view_ix, :] for ob in obs])
                    
                    # Get mask at that viewIndex(rotation)
                    # i.e. not all rotation angles can connect to other vertices on graph
                    # tensor shape (batch_size,)
                    view_ix_mask = torch.tensor(view_index_mask[view_ix], dtype=uint8, device=self.device)

                    # If implementing Ask Agent
                    # ques_asked = ...

                    # Run decoder forward pass
                    decoder_h_curr, _, q_values_rollout_estimate[view_ix], ended_rollout_estimate[view_ix], end_sigmoid, cov_curr = self.model.decode_nav(a_proposed, ques_t, f_proposed, decoder_h, ctx, seq_mask, view_index_mask=view_ix_mask, cov=cov)

                # Reshape q-value estimates back to (batch_size, self.num_viewIndex=36) 
                # .t() expects 2-D tensor
                q_values_rollout_estimate = q_values_rollout_estimate.t()
                ended_rollout_estimate = ended_rollout_estimate.t()

                time_report['decode_frontier'] += time.time() - decode_frontier_start_time
                # -------

                agent_select_start_time = time.time()

                # Select the best view direction determined by expert 
                # tensor shape (batch_size, ), gradient detached
                best_view_ix = self.argmax(-q_values_rollout_estimate)
                # tensor shape (1, batch_size, self.max_macro_action_seq_len)
                best_view_ix_tiled = best_view_ix.view(-1, 1).repeat(1, viewix_actions_map.shape[2]).unsqueeze(0)
                assert best_view_ix_tiled.shape == (1, batch_size, self.max_macro_action_seq_len)

                # Select the macro-action associated with the best view direction
                # tensor shape (batch_size, max_macro_action_seq_len)
                a_t =  viewix_actions_map.gather(0, best_view_ix_tiled).squeeze()
                assert a_t.shape == (batch_size, self.max_macro_action_seq_len)

                time_report['select_agent_macro_action'] += time.time() - agent_select_start_time

            time_report['compute_next_nav_by_feedback'] += time.time() - start_time
            # ---------------------------- 

            # Translate chosen macro-rotations to env actions
            start_time = time.time()
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
            obs = self.env.steps(env_rotations)  # TODO rotate multiple times
            time_report['env_rotation_step'] += time.time() - start_time

            # Get image features after rotation
            # tensor (batch_size, feature size)
            f_t = self.get_feature_variable(obs)

            # Convert to batch_size first for experience update
            # tensor shape (batch_size, 36, self.max_macro_action_seq_len)
            viewix_actions_map = viewix_actions_map.transpose(0, 1)
            # array shape (batch_size, 36)
            view_index_mask = view_index_mask.transpose()
            # array shape (batch_size, 36)
            end_target = end_target.transpose()

            # Convert tensor and arrays to list for saving to json
            # logging only
            q_values_target_list = q_values_target.data.tolist()
            if expert_rollin_bool:
                q_values_rollout_estimate_list = q_values_rollout_estimate.data.tolist()

            # Initialize new experience_batch_t
            # length <= batch_size because some ob may have ended already
            experience_batch_t = []

            # Update experience batch & traj post rotation
            for i, ob in enumerate(obs):
                if not ended[i]:

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
                        'action': a_t[i],  # TODO ending condition

                        # training data - pano sphere
                        # np array shape (varies)
                        'view_index_mask_indices': np.nonzero(view_index_mask[i])[0],
                        # tensors shape (self.num_viewIndex=36, self.max_macro_action_seq_len)
                        'viewix_actions_map': viewix_actions_map[i],
                        # np array shape (varies)
                        'end_target_indices': np.nonzero(end_target[i])[0], 

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

            # Write experience to history buffer
            self.history_buffer.add_experience(experience_batch_t)

            # Append tensors for the whole batch for this time step
            local_buffer.append({
                'a_t': a_t,
                'f_t': f_t,
                # ask, instruction update in Ask Agents
                # end marker TODO
            })

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

                # mark if 'ended' for each ob
                ended[i] = # TODO

            # Early exit if all trajectories in the batch has <end>ed
            if ended.all():
                break
            time_report['save_traj_output'] += time.time() - start_time

            # Increment timestep for the next While loop
            timestep += 1

        time_report['total_rollout_time'] += time.time() - rollout_start_time
        return traj, time_report



# implement explore pano & q_* calculations & determine a_t dimension
# --------------------------
# end problems
# gradient problems