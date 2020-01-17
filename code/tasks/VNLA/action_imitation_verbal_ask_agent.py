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


class ActionImitationVerbalAskAgent(ActionImitationAgent):
    '''
    Same agent as VerbalAskAgent with learned ask mechanism and verbal advisor.
    '''
    
    # inherited ['dont_ask', 'ask', '<start>', '<ignore>']
    # ask_actions = ActionImitationAgent.ask_actions

    def __init__(self, model, hparams, device):
        super(ActionImitationVerbalAskAgent, self).__init__(model, hparams, device)

        self.ask_criterion = nn.CrossEntropyLoss(
            ignore_index = self.ask_actions.index('<ignore>'))

        self.ask_teacher = make_oracle('ask', hparams, self.ask_actions)

        self.advisor = make_oracle(hparams.ask_advisor, hparams.n_subgoal_steps,
            self.nav_actions, self.env_actions)

    @staticmethod
    def n_input_nav_actions():
        # inherited this attribute from NavAgent
        return len(ActionImitationVerbalAskAgent.nav_actions)

    @staticmethod
    def n_output_nav_actions():
        # inherited this attribute from NavAgent
        # cannot output <start> or <ignore> so -2
        return len(ActionImitationVerbalAskAgent.nav_actions) - 2

    @staticmethod
    def n_input_ask_actions():
        # inherited this attribute from ActionImitationAgent
        return len(ActionImitationVerbalAskAgent.ask_actions)

    @staticmethod
    def n_output_ask_actions():
        # inherited this attribute from ActionImitationAgent
        # cannot output <start> or <ignore> so -2
        return len(ActionImitationVerbalAskAgent.ask_actions) - 2
    
    def _setup(self, env, feedback):
        '''called in agent.train()'''
        
        self.nav_feedback = feedback['nav']
        self.ask_feedback = feedback['ask']

        assert self.nav_feedback in self.feedback_options
        assert self.ask_feedback in self.feedback_options

        self.env = env
        self.nav_teacher.add_scans(env.scans)
        self.advisor.add_scans(env.scans)
        self.losses = []
        self.nav_losses = []
        self.ask_losses = []

    def _populate_agent_state_to_obs(self, obs, *args):
        """modify obs in-place (but env is not modified)
        Use cases:
        1. for ask oracle to compute target and reasons
        2. for nav oracle to compute nav target 
        """
        if len(args) > 1:
            # Use case 1
            nav_softmax, queries_unused, traj, ended, time_step = args
            nav_dist = nav_softmax.data.tolist()  # used to compute entropy gap
            for i, ob in enumerate(obs):
                ob['nav_dist'] = nav_dist[i]  
                ob['queries_unused'] = queries_unused[i]
                ob['agent_path'] = traj[i]['agent_path']
                ob['ended'] = ended[i]
                ob['time_step'] = time_step
        elif len(args) == 1:
            # Use case 2
            ended = args 
            for i, ob in enumerate(obs):
                ob['ended'] = ended[i]                     
        
    def _should_ask(self, ended, q):
        return not ended and q == self.ask_actions.index('ask')

    def _compute_loss(self, iter_idx=None):
        '''computed once at the end of every batch'''

        self.loss = self.nav_loss + self.ask_loss

        # .item() get a scalar from torch
        # self.episode_len is hparams.max_episode_length
        iter_loss_avg = self.loss.item() / self.episode_len
        self.losses.append(iter_loss_avg)
        iter_nav_loss_avg = self.nav_loss.item() / self.episode_len
        self.nav_losses.append(iter_nav_loss_avg)
        iter_ask_loss_avg = self.ask_loss.item() / self.episode_len
        self.ask_losses.append(iter_ask_loss_avg)

        # training mode
        if iter_idx: 
            self.SW.add_scalar('train loss per iter', iter_loss_avg, iter_idx)
            self.SW.add_scalar('train nav loss per iter', iter_nav_loss_avg, iter_idx)
            self.SW.add_scalar('train ask loss per iter', iter_ask_loss_avg, iter_idx)
  
    def rollout(self, iter_idx=None):
        
        # time keeping
        time_report = defaultdict(int)
        
        rollout_start_time = time.time()
        start_time = time.time() 

        # Reset environment
        obs = self.env.reset(use_expected_traj_len=self.is_eval)
        batch_size = len(obs)

        # NOTE skipped random ask

        # Index initial command
        seq, seq_mask, seq_lengths = self.make_instruction_batch(obs)

        # Roll-out book keeping
        traj = [{
            # basics
            'scan': ob['scan'],
            'instr_id': ob['instr_id'],
            'agent_path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            # nav related
            'agent_nav': [],
            'agent_nav_logits_tentative': [],
            'agent_nav_softmax_tentative': [],
            'agent_nav_logits_final': [],
            'agent_nav_softmax_final': [],
            'agent_nav_logit_masks': [],
            'teacher_nav': [],
            # asking related
            'agent_ask': [],
            'agent_ask_logits': [],
            'agent_ask_softmax': [],
            'agent_ask_logit_masks': [],
            'teacher_ask': [],
            'teacher_ask_reason': [],
            'subgoals': []
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

        # Initialize nav and ask actions
        a_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * \
            self.nav_actions.index('<start>')
        q_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * \
            self.ask_actions.index('<start>')

        # Traj end markers
        # Whether agent decides to <stop>
        ended = np.array([False] * batch_size)

        # Initialize nav and ask losses to be accumulated across batch
        self.nav_loss = 0
        self.ask_loss = 0

        # Initialize containers for subgoals
        action_subgoals = [[] for _ in range(batch_size)]
        # Initialize counter for subgoal steps taken
        n_subgoal_steps = [0] * batch_size
        # Initialize asking budget
        queries_unused = [ob['max_queries'] for ob in obs]

        # Initialize actions for env (i.e. batch of simulators) to take
        env_action = [None] * batch_size

        # Initialize ^T time budget
        episode_len = max(ob['traj_len'] for ob in obs)

        # time keeping
        time_report['initial_setup'] += time.time() - start_time 

        # loop through time steps
        for time_step in range(episode_len):
            
            # time keeping
            start_time = time.time() 

            # Mask out invalid nav and ask actions
            nav_logit_mask = torch.zeros(batch_size,
                self.n_output_nav_actions(), dtype=torch.bool, device=self.device)
            ask_logit_mask = torch.zeros(batch_size,
                self.n_output_ask_actions(), dtype=torch.bool, device=self.device)
            nav_mask_indices = []
            ask_mask_indices = []
            for i, ob in enumerate(obs):
                # if there are no navigable locations, agent cannot move forward
                if len(ob['navigableLocations']) <= 1:
                    nav_mask_indices.append((i, self.nav_actions.index('forward')))
                # if ask budget has depleted, agent cannot ask
                if queries_unused[i] <= 0:
                    ask_mask_indices.append((i, self.ask_actions.index('ask')))
            nav_logit_mask[list(zip(*nav_mask_indices))] = 1
            ask_logit_mask[list(zip(*ask_mask_indices))] = 1 

            # Get image features
            f_t = self.get_feature_variable(obs)     

            # Ask budget to torch tensor
            b_t = torch.tensor(queries_unused, dtype=torch.long, device=self.device)   

            # time keeping   
            time_report['initial_setup_pertimestep'] += time.time() - start_time 

            # Run first decoder forward pass to compute ask logit
            start_time = time.time()
            _, _, nav_logit_tentative, nav_softmax_tentative, ask_logit, ask_softmax, _ = self.model.decode(
                a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask,
                ask_logit_mask=ask_logit_mask, budget=b_t, cov=cov)
            time_report['decode_tentative'] += time.time() - start_time  

            # logging only
            start_time = time.time() 
            nav_logits_tentative_list = nav_logit_tentative.data.tolist()
            nav_softmax_tentative_list = nav_softmax_tentative.data.tolist()
            nav_logit_mask_list = nav_logit_mask.data.tolist()
            time_report['log_nav_tentative'] += time.time() - start_time            
            # Modify obs in-place for ask prediction
            start_time = time.time()
            self._populate_agent_state_to_obs(obs, nav_softmax_tentative, queries_unused, traj, ended, time_step)
            time_report['pop_state_to_obs_1'] += time.time() - start_time 

            # Query teacher for next ask target action
            start_time = time.time()
            ask_target_list, ask_reason = self.ask_teacher(obs, self.nav_teacher)
            ask_target = torch.tensor(ask_target_list, dtype=torch.long, device=self.device)
            time_report['query_teacher_for_next_ask'] += time.time() - start_time

            # logging only
            start_time = time.time() 
            ask_logits_list = ask_logit.data.tolist()
            ask_softmax_list = ask_softmax.data.tolist()
            ask_logit_masks_list = ask_logit_mask.data.tolist()
            time_report['log_ask'] += time.time() - start_time  

            # Compute ask loss
            start_time = time.time()
            if not self.is_eval:
                self.ask_loss += self.ask_criterion(ask_logit, ask_target)
            time_report['compute_ask_loss'] += time.time() - start_time

            # Determine next ask action by teacher/argmax/sample
            start_time = time.time()
            q_t = self.next_action('ask', ask_logit, ask_target, self.ask_feedback)
            time_report['compute_next_ask_by_feedback'] += time.time() - start_time  

            # Attach subgoals
            start_time = time.time()
            q_t_list = q_t.data.tolist()
            batch_has_asked = False
            verbal_subgoals = [''] * batch_size
            for i in range(batch_size):
                if self._should_ask(ended[i], q_t_list[i]):
                    # Query advisor for subgoal
                    action_subgoals[i], verbal_subgoals[i] = self.advisor(obs[i])  # a list, a string
                    # Prepend subgoal to current instruction as a single str
                    self.env.prepend_instruction(i, verbal_subgoals[i])
                    # Reset subgoal to step index
                    n_subgoal_steps[i] = 0
                    # Decrease ask budget
                    queries_unused[i] -= 1
                    # Mark that an agent in the batch has asked
                    batch_size = True
            time_report['attach_subgoals'] += time.time() - start_time 

            # Encode new instruction if asked
            start_time = time.time()
            if batch_has_asked:
                # Get updated obs since env has been modified
                obs = self.env.get_obs()
                # Make new batch of tokenized instructions
                seq, seq_mask, seq_lengths = self.make_instruction_batch(obs)
                # Re-encode new instructions
                ctx, _ = self.model.encode(seq, seq_lengths)
                # Make new coverage vectors
                if self.coverage_size is not None:
                    # cov has shape(batch_size, max_length, coverage_size)
                    cov = torch.zeros(seq_mask.size(0), seq_mask.size(1), self.coverage_size, dtype=torch.float, device=self.device)
                else:
                    cov = None
            time_report['encode_new_instructions'] += time.time() - start_time 

            # Run second decoder forward pass to compute ask logit          
            start_time = time.time()
            decoder_h, _, nav_logit_final, nav_softmax_final, cov = self.model.decode_nav(
                a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask, ask_logit_mask=None, budget=None, cov=cov)
            time_report['decode_final'] += time.time() - start_time 
     
            # Modify obs in-place for nav teacher to compute next target
            start_time = time.time()
            # we only need to update ob['ended'] for nav_teacher(oracle)
            self._populate_agent_state_to_obs(obs, ended)
            time_report['pop_state_to_obs_2'] += time.time() - start_time   

            # Query teacher for next nav target action   
            start_time = time.time()
            nav_target = self.nav_teacher(obs)
            nav_target = torch.tensor(nav_target, dtype=torch.long, device=self.device)
            time_report['query_teacher_for_next_nav'] += time.time() - start_time

            # logging only
            start_time = time.time()
            nav_logits_final_list = nav_logit_final.data.tolist()
            nav_softmax_final_list = nav_softmax_final.data.tolist()
            nav_target_list = nav_target.data.tolist()
            time_report['logging_nav_final'] += time.time() - start_time 

            # Compute nav loss
            start_time = time.time()
            if not self.is_eval:
                self.nav_loss += self.nav_criterion(nav_logit_final, nav_target)
            time_report['compute_nav_loss'] += time.time() - start_time

            # Determine next nav action by teacher/argmax/sample
            start_time = time.time()
            a_t = self.next_action('nav', nav_logit_final, nav_target, self.nav_feedback)
            time_report['compute_next_nav_by_feedback'] += time.time() - start_time

            # Translate nav actions to env actions for simulators to take
            start_time = time.time()
            a_t_list = a_t.data.tolist()
            for i in range(batch_size):
                # during training, if requested help within last k steps, 
                # do teacher forcing
                if (not self.is_eval) and (n_subgoal_steps[i] < len(action_subgoals[i])):
                    # action_subgoals are SSP actions
                    a_t_list[i] = action_subgoals[i][n_subgoal_steps[i]]
                    n_subgoal_steps[i] += 1
                # Translate to env actions
                env_action[i] = self.nav_teacher.interpret_agent_action(a_t_list[i], obs[i])
            a_t = torch.tensor(a_t_list, dtype=torch.long, device=self.device)
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
                    traj[i]['agent_nav_logits_tentative'].append(nav_logits_tentative_list[i])
                    traj[i]['agent_nav_softmax_tentative'].append(nav_softmax_tentative_list[i])
                    traj[i]['agent_nav_logits_final'].append(nav_logits_final_list[i])
                    traj[i]['agent_nav_softmax_final'].append(nav_softmax_final_list[i])
                    traj[i]['agent_nav_logit_masks'].append(nav_logit_mask_list[i])
                    traj[i]['teacher_nav'].append(nav_target_list[i])
                    # ask related
                    traj[i]['agent_ask'].append(q_t_list[i])
                    traj[i]['agent_ask_logits'].append(ask_logits_list[i])
                    traj[i]['agent_ask_softmax'].append(ask_softmax_list[i])
                    traj[i]['agent_ask_logit_masks'].append(ask_logit_masks_list[i])
                    traj[i]['teacher_ask'].append(ask_target_list[i])
                    traj[i]['teacher_ask_reason'].append(ask_reason[i])
                    traj[i]['subgoals'].append(verbal_subgoals[i])

                    if a_t_list[i] == self.nav_actions.index('<end>') or \
                       time_step >= ob['traj_len'] - 1:
                        ended[i] = True
                assert queries_unused[i] >= 0

            # Early exit if all ended
            if ended.all():
                break
            time_report['save_traj_output'] += time.time() - start_time 

        if not self.is_eval:
            start_time = time.time()
            self._compute_loss(iter_idx)
            time_report['compute_loss_per_batch'] += time.time() - start_time

        time_report['total_rollout_time'] += time.time() - rollout_start_time
        return traj, time_report