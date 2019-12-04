# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import division

import json
import os
import sys
import numpy as np
import random
import time
import math
import copy
from collections import Counter

import torch
import torch.nn as nn
import torch.distributions as D
from torch import optim
import torch.nn.functional as F

from utils import padding_idx
from agent import BaseAgent
from oracle import make_oracle
from ask_agent import AskAgent


class VerbalAskAgent(AskAgent):

    def __init__(self, model, hparams, device):
        super(VerbalAskAgent, self).__init__(model, hparams, device,
                                             should_make_advisor=False)

        assert 'verbal' in hparams.advisor
        if 'easy' in hparams.advisor:
            mode = 'easy'
        elif 'hard' in hparams.advisor:
            mode = 'hard'
        else:
            sys.exit('unknown advisor: %s' % hparams.advisor)

        self.advisor = make_oracle('verbal', hparams.n_subgoal_steps,
                                   self.nav_actions, self.ask_actions,
                                   mode=mode)
        self.hparams = hparams
        self.teacher_interpret = hasattr(hparams, 'teacher_interpret') \
                                 and hparams.teacher_interpret

        self.nav_criterion_bootstrap = nn.CrossEntropyLoss(
            ignore_index=self.nav_actions.index('<ignore>'),
            reduction='none')
        self.ask_criterion_bootstrap = nn.CrossEntropyLoss(
            ignore_index=self.ask_actions.index('<ignore>'),
            reduction='none')

    def prepend_instruction_to_obs(self, obs_k, idx, instr):
        return instr + ' . ' + obs_k[idx]['instruction']

    def rollout(self):
        # Boostrap - rollout with multihead option
        if self.bootstrap:
            return self._rollout_multihead()
        else:
            return self._rollout_single()

    def _rollout_single(self):
        # Reset environment
        obs = self.env.reset(self.is_eval)
        batch_size = len(obs)

        # Sample random ask positions
        if self.random_ask:
            random_ask_positions = [None] * batch_size
            for i, ob in enumerate(obs):
                random_ask_positions[i] = self.random.sample(
                    range(ob['traj_len']), ob['max_queries'])

        # Index initial command
        seq, seq_mask, seq_lengths = self._make_batch(obs)

        # Roll-out bookkeeping
        traj = [{
            'scan': ob['scan'],
            'instr_id': ob['instr_id'],
            'agent_path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            'agent_ask': [],
            'agent_ask_logits': [],  # added
            'agent_ask_softmax': [],  # added
            'teacher_ask': [],
            'teacher_nav': [],  # added
            'teacher_ask_reason': [],
            'agent_nav': [],
            'agent_nav_logits_initial': [],  # added
            'agent_nav_softmax_initial': [],  # added
            'agent_nav_logits_final': [],  # added
            'agent_nav_softmax_final': [],  # added
            'subgoals': [],
            'agent_nav_logit_masks': [],  # added 
            'agent_ask_logit_masks': [],  # added
        } for ob in obs]

        # Encode initial command
        ctx, _ = self.model.encode(seq, seq_lengths)
        decoder_h = None

        if self.coverage_size is not None:
            cov = torch.zeros(seq_mask.size(0), seq_mask.size(1),
                              self.coverage_size,
                              dtype=torch.float, device=self.device)
        else:
            cov = None

        # Initial actions
        a_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * \
            self.nav_actions.index('<start>')
        q_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * \
            self.ask_actions.index('<start>')

        # Whether agent decides to stop
        ended = np.array([False] * batch_size)

        self.nav_loss = 0
        self.ask_loss = 0

        action_subgoals = [[] for _ in range(batch_size)]
        n_subgoal_steps = [0] * batch_size

        env_action = [None] * batch_size
        queries_unused = [ob['max_queries'] for ob in obs]

        episode_len = max(ob['traj_len'] for ob in obs)

        for time_step in range(episode_len):

            # Mask out invalid actions
            nav_logit_mask = torch.zeros(batch_size,
                AskAgent.n_output_nav_actions(), dtype=torch.uint8, device=self.device)
            ask_logit_mask = torch.zeros(batch_size,
                AskAgent.n_output_ask_actions(), dtype=torch.uint8, device=self.device)

            nav_mask_indices = []
            ask_mask_indices = []
            for i, ob in enumerate(obs):
                if len(ob['navigableLocations']) <= 1:
                    nav_mask_indices.append((i, self.nav_actions.index('forward')))

                if queries_unused[i] <= 0:
                    ask_mask_indices.append((i, self.ask_actions.index('ask')))

            nav_logit_mask[list(zip(*nav_mask_indices))] = 1
            ask_logit_mask[list(zip(*ask_mask_indices))] = 1

            # Image features
            f_t = self._feature_variable(obs)

            # Budget features
            b_t = torch.tensor(queries_unused, dtype=torch.long, device=self.device)

            # Run first forward pass to compute ask logit
            _, _, nav_logit, nav_softmax, ask_logit, ask_softmax, _ = self.model.decode(
                a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask,
                ask_logit_mask, budget=b_t, cov=cov)

            # logging only
            nav_logits_initial_list = nav_logit.data.tolist()
            nav_softmax_initial_list = nav_softmax.data.tolist()
            ask_logits_list = ask_logit.data.tolist()
            ask_softmax_list = ask_softmax.data.tolist()
            nav_logit_masks_list = nav_logit_mask.data.tolist()
            ask_logit_masks_list = ask_logit_mask.data.tolist()

            self._populate_agent_state_to_obs(obs, nav_softmax, queries_unused, traj, ended, time_step)

            # Ask teacher for next ask action
            ask_target, ask_reason = self.teacher.next_ask(obs)
            ask_target = torch.tensor(ask_target, dtype=torch.long, device=self.device)
            if not self.is_eval and not (self.random_ask or self.ask_first or self.teacher_ask or self.no_ask):
                self.ask_loss += self.ask_criterion(ask_logit, ask_target)

            # Determine next ask action
            q_t = self._next_action('ask', ask_logit, ask_target, self.ask_feedback)

            # Find which agents have asked and prepend subgoals to their current instructions.
            ask_target_list = ask_target.data.tolist()
            q_t_list = q_t.data.tolist()
            has_asked = False
            verbal_subgoals = [None] * batch_size
            for i in range(batch_size):
                if ask_target_list[i] != self.ask_actions.index('<ignore>'):
                    if self.random_ask:
                        q_t_list[i] = time_step in random_ask_positions[i]
                    elif self.ask_first:
                        q_t_list[i] = int(queries_unused[i] > 0)
                    elif self.teacher_ask:
                        q_t_list[i] = ask_target_list[i]
                    elif self.no_ask:
                        q_t_list[i] = 0

                if self._should_ask(ended[i], q_t_list[i]):
                    # Query advisor for subgoal.
                    action_subgoals[i], verbal_subgoals[i] = self.advisor(obs[i])
                    # Prepend subgoal to the current instruction
                    self.env.prepend_instruction(i, verbal_subgoals[i])
                    # Reset subgoal step index
                    n_subgoal_steps[i] = 0
                    # Decrement queries unused
                    queries_unused[i] -= 1
                    # Mark that some agent has asked
                    has_asked = True

            if has_asked:
                # Update observations
                obs = self.env.get_obs()
                # Make new batch with new instructions
                seq, seq_mask, seq_lengths = self._make_batch(obs)
                # Re-encode with new instructions
                ctx, _ = self.model.encode(seq, seq_lengths)
                # Make new coverage vectors
                if self.coverage_size is not None:
                    # cov has shape(batch_size, max_length, coverage_size)
                    cov = torch.zeros(seq_mask.size(0), seq_mask.size(1), self.coverage_size, dtype=torch.float, device=self.device)
                else:
                    cov = None

            # Run second forward pass to compute nav logit
            # NOTE: q_t and b_t changed since the first forward pass.
            q_t = torch.tensor(q_t_list, dtype=torch.long, device=self.device)
            # b_t = torch.tensor(queries_unused, dtype=torch.long, device=self.device)
            decoder_h, _, nav_logit, nav_softmax, cov = self.model.decode_nav(
                a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask, cov=cov)

            # logging only
            nav_logits_final_list = nav_logit.data.tolist()
            nav_softmax_final_list = nav_softmax.data.tolist()

            # Repopulate agent state
            # NOTE: queries_unused may have changed but it's fine since nav_teacher does not use it!
            self._populate_agent_state_to_obs(obs, nav_softmax, queries_unused,
                traj, ended, time_step)

            # Ask teacher for next nav action
            nav_target = self.teacher.next_nav(obs)
            nav_target = torch.tensor(nav_target, dtype=torch.long, device=self.device)

            # logging only
            nav_target_list = nav_target.data.tolist()

            # Nav loss
            if not self.is_eval:
                self.nav_loss += self.nav_criterion(nav_logit, nav_target)
            # Determine next nav action
            a_t = self._next_action('nav', nav_logit, nav_target, self.nav_feedback)

            # Translate agent action to environment action
            a_t_list = a_t.data.tolist()
            for i in range(batch_size):
                # Conditioned on teacher action during intervention
                # (training only or when teacher_interpret flag is on)
                if (self.teacher_interpret or not self.is_eval) and \
                        n_subgoal_steps[i] < len(action_subgoals[i]):
                    a_t_list[i] = action_subgoals[i][n_subgoal_steps[i]]
                    n_subgoal_steps[i] += 1

                env_action[i] = self.teacher.interpret_agent_action(a_t_list[i], obs[i])

            a_t = torch.tensor(a_t_list, dtype=torch.long, device=self.device)

            # Take nav action
            obs = self.env.step(env_action)

            # Save trajectory output
            ask_target_list = ask_target.data.tolist()
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['agent_path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                    traj[i]['agent_nav'].append(env_action[i])
                    traj[i]['agent_nav_logits_initial'].append(nav_logits_initial_list[i])  # added
                    traj[i]['agent_nav_softmax_initial'].append(nav_softmax_initial_list[i])  #  added
                    traj[i]['agent_nav_logits_final'].append(nav_logits_final_list[i])  #  added
                    traj[i]['agent_nav_softmax_final'].append(nav_softmax_final_list[i])  #  added
                    traj[i]['teacher_nav'].append(nav_target_list[i])  # added
                    traj[i]['teacher_ask'].append(ask_target_list[i])
                    traj[i]['agent_ask'].append(q_t_list[i])
                    traj[i]['agent_ask_logits'].append(ask_logits_list[i])  # added
                    traj[i]['agent_ask_softmax'].append(ask_softmax_list[i])  # added
                    traj[i]['teacher_ask_reason'].append(ask_reason[i])
                    traj[i]['subgoals'].append(verbal_subgoals[i])
                    traj[i]['agent_nav_logit_masks'].append(nav_logit_masks_list[i])  # added
                    traj[i]['agent_ask_logit_masks'].append(ask_logit_masks_list[i])  # added

                    if a_t_list[i] == self.nav_actions.index('<end>') or \
                       time_step >= ob['traj_len'] - 1:
                        ended[i] = True

                assert queries_unused[i] >= 0

            # Early exit if all ended
            if ended.all():
                break

        if not self.is_eval:
            self._compute_loss()

        return traj

    def _rollout_multihead(self):

        # Reset environment
        # obs length=batch_size, each ob is a dictionary
        obs = self.env.reset(self.is_eval)
        batch_size = len(obs)

        # Sample random ask positions
        if self.random_ask:
            random_ask_positions = [None] * batch_size
            for i, ob in enumerate(obs):
                random_ask_positions[i] = self.random.sample(
                   range(ob['traj_len']), ob['max_queries'])

        # Index initial command
        seq, seq_mask, seq_lengths = self._make_batch(obs)

        # Roll-out bookkeeping
        traj = [{
            'scan': ob['scan'],
            'instr_id': ob['instr_id'],
            'agent_path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            'agent_ask': [],
            'agent_ask_logits_heads': [],  # added
            'agent_ask_softmax_heads': [],  # added
            'teacher_ask_heads': [],
            'teacher_nav_heads': [],  # added
            'teacher_ask_reason_heads': [],
            'agent_nav': [],
            'agent_nav_logits_initial_heads': [],  # added
            'agent_nav_softmax_initial_heads': [],  # added
            'agent_nav_logits_final_heads': [],  # added
            'agent_nav_softmax_final_heads': [],  # added
            'subgoals': [],
            'agent_nav_logit_masks': [],  # added
            'agent_ask_logit_masks': [],  # added
            'bootstrap_mask': [],  # Boostrap - track mask
            'active_head': [],  # Boostrap - track heads
            'matching_heads': [], # Boostrap - track heads matching with chosen next actions, list of lists
        } for ob in obs]

        # Encode initial command
        # NOTE bootstrap : the encoder is shared
        ctx, _ = self.model.encode(seq, seq_lengths)
        # NOTE bootstrap : same initial decoder hidden state, None, for all heads
        # decoder_h_heads has len(n_ensemble), each tensor element has shape(batch_size, hidden_size)
        # NOTE bootstrap : each head will preserve its own decoder_h til the end
        decoder_h_heads = [None] * self.n_ensemble

        if self.coverage_size is not None:
            # cov has shape(batch_size, max seq length in the batch, coverage dim)
            cov = torch.zeros(seq_mask.size(0), seq_mask.size(1), self.coverage_size, dtype=torch.float, device=self.device)
        else:
            cov = None

        # Initiate `<start>` actions for a_t and q_t
        # a_t, q_t are renewed between each time step in the for loop below
        # a_t, q_t each has shape(batch_size,)
        a_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * self.nav_actions.index('<start>')
        q_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * self.ask_actions.index('<start>')

        # Whether agent decided to stop
        ended = np.array([False] * batch_size)

        # initiate nav_loss and ask_loss as scalars
        # accumulate through each time step
        self.nav_loss = 0
        self.ask_loss = 0

        # Keep track of subgoals
        action_subgoals = [[] for _ in range(batch_size)]
        n_subgoal_steps = [0] * batch_size

        env_action = [None] * batch_size
        # queries unused will keep decreasing
        # queries_unused length=batch_size
        queries_unused = [ob['max_queries'] for ob in obs]

        episode_len = max(ob['traj_len'] for ob in obs)

        for time_step in range(episode_len):

            # NOTE bootstrap : masks has shape(self.n_ensemble, batch_size)
            masks = np.zeros((self.n_ensemble, batch_size))
            for k in range(self.n_ensemble):
                tot_used = 0
                while tot_used == 0:
                    m = np.random.binomial(1, self.bernoulli_probability, batch_size)
                    tot_used = np.sum(m)
                masks[k] = m

            # Mask out invalid actions
            nav_logit_mask = torch.zeros(batch_size, AskAgent.n_output_nav_actions(), dtype=torch.uint8, device=self.device)
            ask_logit_mask = torch.zeros(batch_size, AskAgent.n_output_ask_actions(), dtype=torch.uint8, device=self.device)

            nav_mask_indices = []
            ask_mask_indices = []
            for i, ob in enumerate(obs):
                if len(ob['navigableLocations']) <= 1:
                    nav_mask_indices.append((i, self.nav_actions.index('forward')))
                if queries_unused[i] <= 0:
                    ask_mask_indices.append((i, self.ask_actions.index('ask')))

            nav_logit_mask[list(zip(*nav_mask_indices))] = 1
            ask_logit_mask[list(zip(*ask_mask_indices))] = 1

            # Image features
            f_t = self._feature_variable(obs)

            # Budget features
            # shape(batch_size, )
            b_t = torch.tensor(queries_unused, dtype=torch.long, device=self.device)

            # Run first forward pass to compute ask logit 
            # NOTE bootstrap : run multiple decoder heads on same input
            # NOTE bootstrap : except decoder_h_heads that is passed from k heads to k heads
            # a_t shape(batch_size,), tensor
            # q_t shape(batch_size,), tensor
            # f_t shape(batch_size, feature size), tensor
            # decoder_h_heads len(n_ensemble), each tensor element has shape(batch_size, hidden_size)
            # ctx shape(batch_size, max_length, ???), tensor
            # seq_mask shape(batch_size, max_length), tensor
            # nav_logit_mask shape(batch_size, n_output_nav_actions), tensor
            # ask_logit_mask shape(batch_size, n_output_ask_actions), tensor
            # b_t shape(batch_size, ), tensor
            # cov shape(batch_size, max seq length in the batch, coverage dim), tensor
            _, _, nav_tentative_logit_heads, nav_tentative_softmax_heads, ask_logit_heads, ask_softmax_heads, _ = self.model.decode(None, a_t, q_t, f_t, decoder_h_heads, ctx, seq_mask, nav_logit_mask, ask_logit_mask, budget=b_t, cov=cov)

            # NOTE logging only
            # cannot call .numpy() unless we put to cpu and detach the tensor first
            nav_tentative_logit_heads_list = torch.stack(nav_tentative_logit_heads, dim=1).cpu().detach().numpy()
            assert nav_tentative_logit_heads_list.shape[0] == batch_size
            assert nav_tentative_logit_heads_list.shape[1] == self.n_ensemble
            nav_tentative_softmax_heads_list = torch.stack(nav_tentative_softmax_heads, dim=1).cpu().detach().numpy()
            ask_logit_heads_list = torch.stack(ask_logit_heads, dim=1).cpu().detach().numpy()
            ask_softmax_heads_list = torch.stack(ask_softmax_heads, dim=1).cpu().detach().numpy()
            nav_logit_masks_list = nav_logit_mask.data.tolist()
            ask_logit_masks_list = ask_logit_mask.data.tolist()

            # Repopulate agent state
            # NOTE bootstrap : get tentative obs for multihead
            # obs length=batch_size, each ob is a dictionary
            # obs_heads_tentative, length=n_ensemble, each element is a list of length=batch_size, each element is a dictionary
            # queries_unused length=batch_size
            obs_heads_tentative = self._populate_agent_state_to_obs_single_to_multihead(obs, nav_tentative_softmax_heads, queries_unused, traj, ended, time_step, self.n_ensemble)

            # Ask teacher for next action
            # NOTE bootstrap : each head has its own ask targets because each head can predict different nav tentative entropies.
            ask_zipped_heads = [self.teacher.next_ask(obs_k) for obs_k in obs_heads_tentative]
            ask_zipped_heads = list(zip(*ask_zipped_heads)) # unzip
            ask_target_heads, ask_reason_heads = ask_zipped_heads[0], ask_zipped_heads[1]
            ask_target_heads = [torch.tensor(ask_target_k, dtype=torch.long, device=self.device) for ask_target_k in ask_target_heads]

            # NOTE logging only
            ask_target_heads_list = torch.stack(ask_target_heads, dim=1).cpu().detach().numpy()
            ask_reason_heads_list = np.array(ask_reason_heads).swapaxes(0,1)

            # Compute ask loss
            # NOTE bootstrap : each head has its own loss before reduced across heads to scalar
            if not self.is_eval and not (self.random_ask or self.ask_first or self.teacher_ask or self.no_ask):
                cnt_ask_loss_heads = [0.0] * self.n_ensemble
                for k in range(self.n_ensemble):
                    ask_loss_k = self.ask_criterion_bootstrap(ask_logit_heads[k], ask_target_heads[k])
                    # masks has shape(self.n_ensemble, batch_size)
                    # mask_k_tensor has shape(batch_size,)
                    mask_k_tensor = torch.tensor(masks[k], dtype=torch.float, device=self.device)
                    assert ask_loss_k.shape == mask_k_tensor.shape
                    # shape(batch_size, )
                    ask_loss_masked_k_full = mask_k_tensor * ask_loss_k
                    # scalar
                    ask_loss_masked_k_full_reduced = torch.sum(ask_loss_masked_k_full/torch.sum(mask_k_tensor))
                    cnt_ask_loss_heads[k] = ask_loss_masked_k_full_reduced
                self.ask_loss += sum(cnt_ask_loss_heads) / self.n_ensemble

            #### Determine next ask action #### 
            # NOTE bootstrap : self._next_action returns pytorch tensors
            q_t_heads = [self._next_action('ask', ask_logit_k, ask_target_k, self.ask_feedback) for ask_logit_k, ask_target_k in zip(ask_logit_heads, ask_target_heads)]

            # Find which agents have asked and prepended subgoals to their current instructions
            # NOTE bootstrap : multihead version
            ask_target_list_heads = [ask_target_k.data.tolist() for ask_target_k in ask_target_heads]
            q_t_list_heads = [q_t_k.data.tolist() for q_t_k in q_t_heads]
            has_asked_heads = [False] * self.n_ensemble
            verbal_subgoals_heads = [[None] * batch_size for _ in range(self.n_ensemble)]
            # NOTE bootstrap : extra steps. these values are changed per head as well
            # seq_mask shape(batch_size, max_length), tensor
            seq_mask_heads = [copy.copy(seq_mask) for _ in range(self.n_ensemble)]
            # action_subgoals is a list of length batch_size, each a []
            action_subgoals_heads = [copy.copy(action_subgoals) for _ in range(self.n_ensemble)]
            # n_subgoal_steps is a list of length batch_size, each int
            n_subgoal_steps_heads = [copy.copy(n_subgoal_steps) for _ in range(self.n_ensemble)]
            # queries_unused is a list of length batch_size, each int
            queries_unused_heads = [copy.copy(queries_unused) for _ in range(self.n_ensemble)]
            # ctx shape(batch_size, max_length, ???), tensor 
            ctx_heads = [copy.copy(ctx) for _ in range(self.n_ensemble)]
            # cov shape(batch_size, max seq length in the batch, coverage dim), tensor
            cov_heads = [copy.copy(cov) for _ in range(self.n_ensemble)]

            for k in range(self.n_ensemble):
                for i in range(batch_size):
                    # actual ask action can change if using baseline meta-algo
                    if ask_target_list_heads[k][i] != self.ask_actions.index('<ignore>'):
                        if self.random_ask:
                            q_t_list_heads[k][i] = time_step in random_ask_positions[i] # same for all heads
                        elif self.ask_first:
                            q_t_list_heads[k][i] = int(queries_unused[i] > 0)
                        elif self.teacher_ask:
                            q_t_list_heads[k][i] = ask_target_list_heads[k][i]
                        elif self.no_ask:
                            q_t_list_heads[k][i] = 0
                    
                    # if ask, get subgoal, prepend it
                    if self._should_ask(ended[i], q_t_list_heads[k][i]):
                        # Query advisor for subgoal.
                        action_subgoals_heads[k][i], verbal_subgoals_heads[k][i] = self.advisor(obs_heads_tentative[k][i])
                        # Prepend subgoal to the current instruction in obs
                        # NOTE bootstrap : modify obs_head instead of env
                        # NOTE bootstrap : modify env after voting below
                        obs_heads_tentative[k][i]['instruction'] = self.prepend_instruction_to_obs(obs_heads_tentative[k], i, verbal_subgoals_heads[k][i])
                        # Reset subgoal step index
                        n_subgoal_steps_heads[k][i] = 0
                        # Decrement queries unused
                        queries_unused_heads[k][i] -= 1
                        # Mark that some agent has asked
                        has_asked_heads[k] = True   

                # if we had asked in the batch, re-encode the modified language instructions
                if has_asked_heads[k]:
                    # Make new batch with new instructions
                    seq, seq_mask_heads[k], seq_lengths = self._make_batch(obs_heads_tentative[k])
                    # Re-encode with new instructions
                    ctx_heads[k], _ = self.model.encode(seq, seq_lengths)
                    # Make new coverage vectors
                    if self.coverage_size is not None:
                        # cov has shape(batch_size, max_length, coverage_size)
                        cov_heads[k] = torch.zeros(seq_mask_heads[k].size(0), seq_mask_heads[k].size(1), self.coverage_size, dtype=torch.float, device=self.device)
                    else:
                        cov_heads[k] = None

            # Run second forward pass to compute nav logit
            #  NOTE: q_t and b_t changed since the first forward pass.
            q_t_heads = [torch.tensor(q_t_list, dtype=torch.long, device=self.device) for q_t_list in q_t_list_heads]
            # inputs:
            # a_t -- stays the same. action from last timestep
            # q_t_heads -- changed. updated from asking and multihead
            # f_t -- stays the same. picture frame
            # decoder_h_heads -- multihead version. from last time step
            # ctx_heads -- changed. depending on whether we have asked. 
            # seq_mask_heads -- changed. depending on whether we have asked. 
            # nav_logit_mask -- stays the same. the same mask for every timestep
            # cov_heads -- changed. depending on whether we have asked.
            decoder_h_heads, _, nav_final_logit_heads, nav_final_softmax_heads, cov_heads = self.model.decode_nav(None, a_t, q_t_heads, f_t, decoder_h_heads, ctx_heads, seq_mask_heads, nav_logit_mask, cov=cov_heads)

            # NOTE logging only
            nav_final_logit_heads_list = torch.stack(nav_final_logit_heads, dim=1).cpu().detach().numpy()
            assert nav_final_logit_heads_list.shape[0] == batch_size
            assert nav_final_logit_heads_list.shape[1] == self.n_ensemble
            nav_final_softmax_heads_list = torch.stack(nav_final_softmax_heads, dim=1).cpu().detach().numpy()

            # Repopulate agent state
            # NOTE: queries_unused may have changed but it's fine since nav_teacher does not use it!
            obs_heads_final = self._populate_agent_state_to_obs_multi_to_multihead(obs_heads_tentative, nav_final_softmax_heads, queries_unused_heads, traj, ended, time_step, self.n_ensemble)            
            # Ask teacher for next nav action
            nav_target_heads = [self.teacher.next_nav(obs_k) for obs_k in obs_heads_final]
            nav_target_heads = [torch.tensor(nav_target_k, dtype=torch.long, device=self.device) for nav_target_k in nav_target_heads]

            # NOTE logging only
            nav_target_heads_list = torch.stack(nav_target_heads, dim=1).cpu().detach().numpy()

            # Nav loss
            # NOTE bootstrap : apply masking
            if not self.is_eval:
                cnt_nav_loss_heads = [0.0] * self.n_ensemble
                for k in range(self.n_ensemble):
                    nav_loss_k = self.nav_criterion_bootstrap(nav_final_logit_heads[k], nav_target_heads[k])
                    # mask_k_tensor has shape(batch_size,)
                    mask_k_tensor = torch.tensor(masks[k], dtype=torch.float, device=self.device)
                    assert nav_loss_k.shape == mask_k_tensor.shape
                    # shape(batch_size, )
                    nav_loss_masked_k_full = mask_k_tensor * nav_loss_k
                    # scalar
                    nav_loss_masked_k_reduced = torch.sum(nav_loss_masked_k_full/torch.sum(mask_k_tensor))
                    # accumulate across heads
                    cnt_nav_loss_heads[k] = nav_loss_masked_k_reduced
                self.nav_loss += sum(cnt_nav_loss_heads)/self.n_ensemble

            #### Determine next nav action #### 
            a_t_heads = [self._next_action('nav', nav_final_logit_heads[k], nav_target_heads[k], self.nav_feedback) for k in range(self.n_ensemble)] 

            # meta algo adjustments to a_t
            a_t_list_heads = [a_t_k.data.tolist() for a_t_k in a_t_heads]
            for k in range(self.n_ensemble):
                for i in range(batch_size):
                    # Conditioned on teacher action during intervention
                    # (training only or when teacher_interpret flag is on)
                    if (self.teacher_interpret or not self.is_eval) and \
                        n_subgoal_steps_heads[k][i] < len(action_subgoals_heads[k][i]):  
                        a_t_list_heads[k][i] = action_subgoals_heads[k][i][n_subgoal_steps_heads[k][i]]
                        n_subgoal_steps_heads[k][i] += 1

                # NOTE bootstrap : do not convert to tensor until after voting
                # a_t_heads[k] = torch.tensor(a_t_list_heads[k], dtype=torch.long, device=self.device)
            
            #### Vote/Sample for next step ####
            heads_ref = np.zeros(batch_size)
            matching_heads = [list() for _ in range(batch_size)]
            if self.bootstrap_majority_vote:
                
                majority_vote = [tuple() for i in range(batch_size)]
                # shape(n_ensemble, batch_size, 2)
                q_a_pairs = [None] * self.n_ensemble
                for k in range(self.n_ensemble):
                    # q_t_list_heads[k] np array shape(batch_size,)
                    # a_t_list_heads[k] np array shape(batch_size,)
                    q_a_pairs[k] = np.stack([q_t_list_heads[k], a_t_list_heads[k]], axis=1)
                # stack to shape(batch_size, n_ensemble, 2)
                q_a_pairs = np.stack(q_a_pairs, axis=1)
                assert q_a_pairs.shape[0] == batch_size
                assert q_a_pairs.shape[1] == self.n_ensemble
                
                # for each datapoint in the batch, let the k heads vote
                for i in range(batch_size):
                    votes = q_a_pairs[i]
                    votes = [tuple(vote) for vote in votes]
                    majority_vote[i] = Counter(votes).most_common(1)[0][0]
                    # sample a final-vote-matching head to reference
                    for k in range(self.n_ensemble):
                        if majority_vote[i] == votes[k]:
                            matching_heads[i].append(k)
                    heads_ref[i] = np.random.choice(matching_heads, 1)

                # for LSTM decoder input
                a_t_list = [a for (q, a) in majority_vote]
                q_t_list = [q for (q, a) in majority_vote]
                a_t = torch.tensor(a_t_list, dtype=torch.long, device=self.device)
                q_t = torch.tensor(q_t_list, dtype=torch.long, device=self.device)
                assert a_t.shape[0] == q_t.shape[0] == batch_size
                
            else:
                # sample a head for each datapt in the batch
                # shape(batch_size, )
                heads_ref = self.random_state.choice(self.n_ensemble, batch_size)
                a_t_list_heads = np.array(a_t_list_heads).swapaxes(0, 1)
                assert a_t_list_heads.shape[0] == batch_size
                a_t_list = np.stack([vals[head] for vals, head in zip(a_t_list_heads, heads_ref)]) # needed
                a_t = torch.tensor(a_t_list, dtype=torch.long, device=self.device)
                q_t_list_heads = np.array(q_t_list_heads).swapaxes(0, 1)
                assert q_t_list_heads.shape[0] == batch_size
                q_t_list = np.stack([vals[head] for vals, head in zip(q_t_list_heads, heads_ref)])
                q_t = torch.tensor(q_t_list, dtype=torch.long, device=self.device)

            # prepare for LSTM decoder input in next timestep
            # convert to shape(batch_size, n_ensemble, *feature sizes)
            cov_heads = torch.stack(cov_heads, dim=1)
            assert cov_heads.shape[0] == batch_size
            cov = torch.stack([vals[head] for vals, head in zip(cov_heads, heads_ref)])
            ctx_heads = torch.stack(ctx_heads, dim=1)
            assert ctx_heads.shape[0] == batch_size
            ctx = torch.stack([vals[head] for vals, head in zip(ctx_heads, heads_ref)])
            seq_mask_heads = torch.stack(seq_mask_heads, dim=1)
            assert seq_mask_heads.shape[0] == batch_size
            seq_mask = torch.stack([vals[head] for vals, head in zip(seq_mask_heads, heads_ref)])

            # prepare for meta-level algo in next timestep
            # original queries_unused_heads - len=n_ensemble, each len=batch_size.
            queries_unused_heads = np.array(queries_unused_heads).swapaxes(0, 1)
            assert queries_unused_heads.shape[0] == batch_size
            queries_unused = np.stack([vals[head] for vals, head in zip(queries_unused_heads, heads_ref)])
            # original n_subgoal_steps_heads - len=n_ensemble, each len=batch_size
            n_subgoal_steps_heads = np.array(n_subgoal_steps_heads).swapaxes(0, 1)
            assert n_subgoal_steps_heads.shape[0] == batch_size
            n_subgoal_steps = np.stack([vals[head] for vals, head in zip(n_subgoal_steps_heads, heads_ref)])
            # original action_subgoals_heads - len=n_ensemble, each len=batch_size, each len=hparams.n_subgoal_steps
            action_subgoals_heads = np.array(action_subgoals_heads).swapaxes(0, 1)
            assert action_subgoals_heads.shape[0] == batch_size
            action_subgoals = np.stack([vals[head] for vals, head in zip(action_subgoals_heads, heads_ref)])
            # original verbal_subgoals_heads - len=n_ensemble, each len=batch_size
            verbal_subgoals_heads = np.array(verbal_subgoals_heads).swapaxes(0, 1)
            assert verbal_subgoals_heads.shape[0] == batch_size
            verbal_subgoals = np.stack([vals[head] for vals, head in zip(verbal_subgoals_heads, heads_ref)])
            # obs_heads_final[k][i] = {"nav_dist":..., "queries_unused":..., "agent_path":...., "ended":...., ...}
            obs = []
            for i in range(batch_size):
                obs.append(obs_heads_final[heads_ref[i]][i])


            # write subgoal instructions back to env
            for i in range(batch_size):
                if self._should_ask(ended[i], q_t_list[i]):
                    # Prepend subgoal to the current instruction in env
                    self.env.prepend_instruction(i, verbal_subgoals[i])

            # Loop through batch and write env_action[i]
            for i in range(batch_size):
                env_action[i] = self.teacher.interpret_agent_action(a_t_list[i], obs[i])

            # Take nav action
            obs = self.env.step(env_action)

            # Save trajectory output
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['agent_path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                    traj[i]['agent_nav'].append(env_action[i])
                    traj[i]['agent_nav_logits_initial_heads'].append(nav_tentative_logit_heads_list[i])  # added
                    traj[i]['agent_nav_softmax_initial_heads'].append(nav_tentative_softmax_heads_list[i])  # added
                    traj[i]['agent_nav_logits_final_heads'].append(nav_final_logit_heads_list[i])  # added
                    traj[i]['agent_nav_softmax_final_heads'].append(nav_final_softmax_heads_list[i])  # added
                    traj[i]['teacher_nav_heads'].append(nav_target_heads_list[i])  # added
                    traj[i]['teacher_ask_heads'].append(ask_target_heads_list[i])  # modified
                    traj[i]['agent_ask'].append(q_t_list[i])
                    traj[i]['agent_ask_logits_heads'].append(ask_logit_heads_list[i])  # added
                    traj[i]['agent_ask_softmax_heads'].append(ask_softmax_heads_list[i])  # added
                    traj[i]['teacher_ask_reason_heads'].append(ask_reason_heads_list[i]) # added 
                    traj[i]['subgoals'].append(verbal_subgoals[i])
                    traj[i]['agent_nav_logit_masks'].append(nav_logit_masks_list[i])  # added
                    traj[i]['agent_ask_logit_masks'].append(ask_logit_masks_list[i])  # added
                    traj[i]['bootstrap_mask'].append(masks[:, i])  # added. masks has shape(n_ensemble, batch size)
                    if not self.bootstrap_majority_vote:
                        traj[i]['active_head'].append(heads_ref[i])  # added
                        traj[i]['matching_heads'].append(matching_heads[i])  # added
                    if a_t_list[i] == self.nav_actions.index('<end>') or \
                       time_step >= ob['traj_len'] - 1:
                        ended[i] = True

                assert queries_unused[i] >= 0 
            
            # Early exit if all ended
            if ended.all():
                break

        if not self.is_eval:
            # Bootstrap - should remain the same because self.ask_loss and self.nav_loss are reduced to scalars
            self._compute_loss()

        return traj
