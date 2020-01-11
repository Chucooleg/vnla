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
from action_imitation_agent import ActionImitationAgent
from oracle import make_oracle


class ActionImitationNoAskAgent(ActionImitationAgent):
    '''
    Imitation Agent with NO asking mechanism.
    '''

    def __init__(self, model, hparams, device):
        super(ActionImitationNoAskAgent, self).__init__(model, hparams, device)

    @staticmethod
    def n_input_nav_actions():
        # inherited this attribute from NavAgent
        return len(ActionImitationNoAskAgent.nav_actions)

    @staticmethod
    def n_output_nav_actions():
        # inherited this attribute from NavAgent
        # cannot output <start> or <ignore> so -2
        return len(ActionImitationNoAskAgent.nav_actions) - 2

    @staticmethod
    def n_input_ask_actions():
        # inherited this attribute from ActionImitationAgent
        return len(ActionImitationNoAskAgent.ask_actions)

    @staticmethod
    def n_output_ask_actions():
        # inherited this attribute from ActionImitationAgent
        # cannot output <start> or <ignore> so - 2
        return len(ActionImitationNoAskAgent.ask_actions) - 2

    def _setup(self, env, feedback):
        '''called in train() in parent class'''
        
        self.nav_feedback = feedback['nav']
        assert self.nav_feedback in self.feedback_options

        self.env = env
        self.nav_teacher.add_scans(env.scans)
        self.losses = []
        self.nav_losses = [] 

    def _populate_agent_state_to_obs(self, obs, ended):
        """modify obs in-place (but env is not modified)
        for nav oracle to compute nav target 
        """
        for i, ob in enumerate(obs):
            ob['ended'] = ended[i] 

    def _compute_loss(self, iter_idx=None):
        '''computed once at the end of every batch'''

        self.loss = self.nav_loss
        # + self.ask_loss for ask agents

        # .item() get a scalar from torch
        # self.episode_len is hparams.max_episode_length
        iter_loss_avg = self.loss.item() / self.episode_len
        self.losses.append(iter_loss_avg)
        iter_nav_loss_avg = self.nav_loss.item() / self.episode_len
        self.nav_losses.append(iter_nav_loss_avg)

        # training mode
        if iter_idx: 
            self.SW.add_scalar('train loss per iter', iter_loss_avg, iter_idx)
            self.SW.add_scalar('train nav loss per iter', iter_nav_loss_avg, iter_idx)  

    def rollout(self, iter_idx=None):

        # time keeping
        time_report = defaultdict(int)

        rollout_start_time = time.time()
        start_time = time.time() 

        # Reset environment
        obs = self.env.reset(self.is_eval)
        batch_size = len(obs)

        # Index initial command
        seq, seq_mask, seq_lengths = self.make_instruction_batch(obs)

        # Roll-out book keeping
        traj = [{
            # indexing
            'scan': ob['scan'],
            'instr_id': ob['instr_id'],

            # trajectory
            'agent_path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],

            # nav related
            'agent_nav': [],
            'agent_nav_logits': [],
            'agent_nav_softmax': [],
            'agent_nav_logit_masks': [],
            'teacher_nav': []
        } for ob in obs]

        # Encode initial command
        ctx, _ = self.model.encode(seq, seq_lengths)

        # Initialize decoder hidden state
        decoder_h = None

        # Optional coverage vector
        if self.coverage_size is not None:
            cov = torch.zeros(seq_mask.size(0), seq_mask.size(1),
                              self.coverage_size,
                              dtype=torch.float, device=self.device)
        else:
            cov = None

        # Initialize nav action
        a_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * \
            self.nav_actions.index('<start>')

        # Initialize dummy ask action
        # This tensor is always dont_ask because agent doesn't ask
        q_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * \
            self.ask_actions.index('dont_ask')

        # Trajectory 'end' markers
        # Whether agent has decided to <stop>
        ended = np.array([False] * batch_size)

        # Initialize nav loss to be accumulated across batch
        self.nav_loss = 0

        # Initialize actions for env (i.e. batch of simulators) to take
        # [(0,1,0), (0,0,1), ...]
        env_action = [None] * batch_size

        # Initialize ^T time budget
        # Precomputed per trajectory when Env() was initialized
        episode_len = max(ob['traj_len'] for ob in obs)

        # time keeping
        time_report['initial_setup'] += time.time() - start_time         

        # loop through time steps
        for time_step in range(episode_len):
            
            # time keeping
            start_time = time.time() 

            # Mask out invalid nav actions
            nav_logit_mask = torch.zeros(batch_size,
                self.n_output_nav_actions(), dtype=torch.bool, device=self.device)
            nav_mask_indices = []
            for i, ob in enumerate(obs):
                # if there are no navigable locations, agent cannot move forward
                if len(ob['navigableLocations']) <= 1:
                    nav_mask_indices.append((i, self.nav_actions.index('forward')))
            nav_logit_mask[list(zip(*nav_mask_indices))] = 1

            # Get image features
            f_t = self.get_feature_variable(obs)
 
            # Run decoder forward pass
            start_time = time.time() 
            decoder_h, _, nav_logit, nav_softmax, cov = self.model.decode_nav(
                a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask,
                ask_logit_mask=None, budget=None, cov=cov)
            time_report['decode_nav'] += time.time() - start_time

            # Modify obs in-place for nav teacher to compute next target
            start_time = time.time()
            # we only need to update ob['ended'] for nav_teacher(oracle)
            self._populate_agent_state_to_obs(obs, ended)
            time_report['pop_state_to_obs'] += time.time() - start_time
                        
            # Query teacher for next nav target action
            start_time = time.time()
            nav_target = self.nav_teacher(obs)
            nav_target = torch.tensor(nav_target, dtype=torch.long, device=self.device)
            time_report['query_teacher_for_next_nav'] += time.time() - start_time

            # logging only
            start_time = time.time() 
            nav_logits_list = nav_logit.data.tolist()
            nav_softmax_list = nav_softmax.data.tolist()
            nav_logit_mask_list = nav_logit_mask.data.tolist()
            nav_target_list = nav_target.data.tolist()
            time_report['log_nav'] += time.time() - start_time

            # Compute nav loss
            start_time = time.time()
            if not self.is_eval:
                self.nav_loss += self.nav_criterion(nav_logit, nav_target)
            time_report['compute_nav_loss'] += time.time() - start_time

            # Determine next nav action by teacher/argmax/sample
            start_time = time.time()
            a_t = self.next_action('nav', nav_logit, nav_target, self.nav_feedback)
            time_report['compute_next_nav_by_feedback'] += time.time() - start_time

            # Translate nav actions to env actions for simulators to take
            start_time = time.time()
            a_t_list = a_t.data.tolist()
            for i in range(batch_size):
                # Translate to env actions
                env_action[i] = self.nav_teacher.interpret_agent_action(a_t_list[i], obs[i])
            time_report['translate_to_env_action'] += time.time() - start_time

            # Simulator take nav action
            start_time = time.time()
            obs = self.env.step(env_action)
            time_report['env_step'] += time.time() - start_time

            # Save trajectory output
            start_time = time.time()
            for i, ob in enumerate(obs):
                if not ended[i]:
                    # basics
                    traj[i]['agent_path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                    # nav related
                    traj[i]['agent_nav'].append(env_action[i])
                    traj[i]['agent_nav_logits'].append(nav_logits_list[i])
                    traj[i]['agent_nav_softmax'].append(nav_softmax_list[i])
                    traj[i]['agent_nav_logit_masks'].append(nav_logit_mask_list[i])
                    traj[i]['teacher_nav'].append(nav_target_list[i])

                    if a_t_list[i] == self.nav_actions.index('<end>') or \
                       time_step >= ob['traj_len'] - 1:
                        ended[i] = True
                
            # Early exit if all ended
            if ended.all():
                break
            time_report['save_traj_output'] += time.time() - start_time

        # Compute per batch loss, append to *.losses
        if not self.is_eval:
            start_time = time.time()
            self._compute_loss(iter_idx)
            time_report['compute_loss_per_batch'] += time.time() - start_time

        time_report['total_rollout_time'] += time.time() - rollout_start_time
        return traj, time_report           