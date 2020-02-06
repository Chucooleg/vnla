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

import torch
import torch.nn as nn
import torch.distributions as D
from torch import optim
import torch.nn.functional as F

from utils import padding_idx
from agent import BaseAgent
from oracle import make_oracle

from tensorboardX import SummaryWriter


class AskAgent(BaseAgent):

    nav_actions = ['left', 'right', 'up', 'down',
                   'forward', '<end>', '<start>', '<ignore>']
    env_actions = [
        (0,-1, 0), # left
        (0, 1, 0), # right
        (0, 0, 1), # up
        (0, 0,-1), # down
        (1, 0, 0), # forward
        (0, 0, 0), # <end>
        (0, 0, 0), # <start>
        (0, 0, 0)  # <ignore>
    ]
    ask_actions = ['dont_ask', 'ask', '<start>', '<ignore>']
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, model, hparams, room_types, device, should_make_advisor=True):
        super(AskAgent, self).__init__(hparams.seed)
        self.model = model
        self.episode_len = hparams.max_episode_length
        self.nav_criterion = nn.CrossEntropyLoss(
            ignore_index = self.nav_actions.index('<ignore>'))
        self.ask_criterion = nn.CrossEntropyLoss(
            ignore_index = self.ask_actions.index('<ignore>'))

        self.teacher = make_oracle('next_optimal', hparams, self.nav_actions,
            self.env_actions, self.ask_actions)
        if should_make_advisor:
            self.advisor = make_oracle(hparams.advisor, hparams.n_subgoal_steps,
                self.nav_actions, self.env_actions)

        self.device = device

        self.random = random
        self.random.seed(hparams.seed)

        self.random_ask = hparams.random_ask if hasattr(hparams, 'random_ask') else 0
        self.ask_first = hparams.ask_first if hasattr(hparams, 'ask_first') else 0
        self.teacher_ask = hparams.teacher_ask if hasattr(hparams, 'teacher_ask') else 0
        self.no_ask = hparams.no_ask if hasattr(hparams, 'no_ask') else 0

        self.max_input_length = hparams.max_input_length
        self.n_subgoal_steps  = hparams.n_subgoal_steps

        self.coverage_size = hparams.coverage_size if hasattr(hparams, 'coverage_size') else None

        # semantics update
        self.with_semantics = hasattr(hparams, 'with_semantics') \
                                 and hparams.with_semantics
        if self.with_semantics:
            self.room_types = room_types
            if hparams.room_cheat:
                self.room_classifier = make_oracle('curr_room_type', room_types)
            else:
                raise ValueError('learned room classifier not implemented yet.')

        # blind fold
        self.blind_fold = hasattr(hparams, 'blind_fold') \
                                 and hparams.blind_fold
        self.img_feature_size = hparams.img_feature_size

        # self.SW = SummaryWriter(hparams.tensorboard_dir, flush_secs=30)

    @staticmethod
    def n_input_nav_actions():
        return len(AskAgent.nav_actions)

    @staticmethod
    def n_output_nav_actions():
        return len(AskAgent.nav_actions) - 2

    @staticmethod
    def n_input_ask_actions():
        return len(AskAgent.ask_actions)

    @staticmethod
    def n_output_ask_actions():
        return len(AskAgent.ask_actions) - 2

    def _make_batch(self, obs):
        ''' Make a variable for a batch of input instructions. '''
        seq_tensor = np.array([self.env.encode(ob['instruction']) for ob in obs])

        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)

        max_length = max(seq_lengths)
        assert max_length <= self.max_input_length

        seq_tensor = torch.from_numpy(seq_tensor).long().to(self.device)[:, :max_length]
        seq_lengths = torch.from_numpy(seq_lengths).long().to(self.device)

        # for len_val in seq_lengths:
        #     if len_val <= 0:
        #         import pdb; pdb.set_trace()

        mask = (seq_tensor == padding_idx)

        return seq_tensor, mask, seq_lengths

    def _feature_variable(self, obs):
        ''' Make a variable for a batch of precomputed image features. '''
        feature_size = obs[0]['feature'].shape[0]
        features = np.empty((len(obs),feature_size), dtype=np.float32)
        for i,ob in enumerate(obs):
            features[i,:] = ob['feature']
        return torch.from_numpy(features).to(self.device)

    def _argmax(self, logit):
        return logit.max(1)[1].detach()

    def _sample(self, logit):
        prob = F.softmax(logit, dim=1).contiguous()

        # Weird bug with torch.multinomial: it samples even zero-prob actions.
        while True:
            sample = torch.multinomial(prob, 1, replacement=True).view(-1)
            is_good = True
            for i in range(logit.size(0)):
                if logit[i, sample[i].item()].item() == -float('inf'):
                    is_good = False
                    break
            if is_good:
                break

        return sample

    def _next_action(self, name, logit, target, feedback):
        ''' Determine the next action to take based on the training algorithm. '''

        if feedback == 'teacher':
            return target
        if feedback == 'argmax':
            return self._argmax(logit)
        if feedback == 'sample':
            return self._sample(logit)
        sys.exit('Invalid feedback option')

    def _populate_agent_state_to_obs(self, obs, *args):
        """modify obs tentatively for asking policy (but environment is not modified)"""
        nav_softmax, queries_unused, traj, ended, time_step = args
        nav_dist = nav_softmax.data.tolist()
        for i, ob in enumerate(obs):
            ob['nav_dist'] = nav_dist[i]
            ob['queries_unused'] = queries_unused[i]
            ob['agent_path'] = traj[i]['agent_path']
            ob['ended'] = ended[i]
            ob['time_step'] = time_step

    def _should_ask(self, ended, q):
        return not ended and q == self.ask_actions.index('ask')

    def rollout(self, iter_idx=None):
        # iter_idx indicates which batch we are at over the whole training process
        # Reset environment
        obs = self.env.reset(self.is_eval)
        batch_size = len(obs)

        # Sample random ask positions
        if self.random_ask:
            random_ask_positions = [None] * batch_size
            for i, ob in enumerate(obs):
                random_ask_positions[i] = self.random.sample(
                    range(ob['traj_len']), ob['max_queries'])

        seq, seq_mask, seq_lengths = self._make_batch(obs)

        # History
        traj = [{
            'scan': ob['scan'],
            'instr_id': ob['instr_id'],
            'agent_path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            'agent_ask': [],
            'teacher_ask': [],
            'teacher_ask_reason': [],
            'agent_nav' : [],
            'subgoals'  : []
        } for ob in obs]

        # Encode initial command
        ctx, _ = self.model.encode(seq, seq_lengths)
        decoder_h = None

        # Coverage vector
        if self.coverage_size is not None:
            cov = torch.zeros(seq_mask.size(0), seq_mask.size(1), self.coverage_size,
                dtype=torch.float, device=self.device)
        else:
            cov = None

        # Initial actions
        a_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * \
            self.nav_actions.index('<start>')
        q_t = torch.ones(batch_size, dtype=torch.long, device=self.device) * \
            self.ask_actions.index('<start>')

        # Whether agent has decided to stop
        ended = np.array([False] * batch_size)

        self.nav_loss = 0
        self.ask_loss = 0

        subgoals = [[] for _ in range(batch_size)]
        n_subgoal_steps = [0] * batch_size
        teacher_actions = [None] * batch_size

        env_action = [None] * batch_size
        queries_unused = [ob['max_queries'] for ob in obs]

        episode_len = max(ob['traj_len'] for ob in obs)

        for time_step in range(episode_len):

            nav_logit_mask = torch.zeros(batch_size,
                AskAgent.n_output_nav_actions(), dtype=torch.bool, device=self.device)
            ask_logit_mask = torch.zeros(batch_size,
                AskAgent.n_output_ask_actions(), dtype=torch.bool, device=self.device)

            # Mask invalid actions
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

            # Take a decoding step
            decoder_h, alpha, nav_logit, nav_softmax, ask_logit, cov = \
                self.model.decode(a_t, q_t, f_t, decoder_h, ctx, seq_mask,
                    nav_logit_mask, ask_logit_mask, budget=b_t, cov=cov)

            self._populate_agent_state_to_obs(obs, nav_softmax, queries_unused,
                traj, ended, time_step)

            # Query teacher for next actions
            nav_target, ask_target, ask_reason = self.teacher(obs)

            # Nav loss
            nav_target = torch.tensor(nav_target, dtype=torch.long, device=self.device)
            if not self.is_eval:
                self.nav_loss += self.nav_criterion(nav_logit, nav_target)

            # Ask loss
            ask_target = torch.tensor(ask_target, dtype=torch.long, device=self.device)
            if not self.is_eval and not (self.random_ask or self.ask_first or self.teacher_ask or self.no_ask):
                self.ask_loss += self.ask_criterion(ask_logit, ask_target)

            # Determine next actions
            a_t = self._next_action('nav', nav_logit, nav_target, self.nav_feedback)
            q_t = self._next_action('ask', ask_logit, ask_target, self.ask_feedback)

            nav_target_list = nav_target.data.tolist()
            ask_target_list = ask_target.data.tolist()
            a_t_list = a_t.data.tolist()
            q_t_list = q_t.data.tolist()

            for i in range(batch_size):
                # Change ask action according to policy
                if ask_target_list[i] != self.ask_actions.index('<ignore>'):
                    if self.random_ask:
                        q_t_list[i] = time_step in random_ask_positions[i]
                    elif self.ask_first:
                        q_t_list[i] = int(queries_unused[i] > 0)
                    elif self.teacher_ask:
                        q_t_list[i] = ask_target_list[i]
                    elif self.no_ask:
                        q_t_list[i] = 0

                # If ask
                if self._should_ask(ended[i], q_t_list[i]):
                    # Query advisor for subgoals
                    subgoals[i] = self.advisor(obs[i])
                    # Reset subgoal step index
                    n_subgoal_steps[i] = 0
                    # Decrement queries unused
                    queries_unused[i] -= 1

                # Direct advisor: If still executing a subgoal, overwrite agent's
                #   decision by advisor's decision.
                if n_subgoal_steps[i] < len(subgoals[i]):
                    a_t_list[i] = subgoals[i][n_subgoal_steps[i]]
                    n_subgoal_steps[i] += 1

                # Map the agent's action back to the simulator's action space.
                env_action[i] = self.teacher.interpret_agent_action(a_t_list[i], obs[i])

            a_t = torch.tensor(a_t_list, dtype=torch.long, device=self.device)
            q_t = torch.tensor(q_t_list, dtype=torch.long, device=self.device)

            # Execute nav actions
            obs = self.env.step(env_action)

            # Update history
            ask_target_list = ask_target.data.tolist()
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['agent_path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                    traj[i]['agent_nav'].append(env_action[i])
                    traj[i]['teacher_ask'].append(ask_target_list[i])
                    traj[i]['agent_ask'].append(q_t_list[i])
                    traj[i]['teacher_ask_reason'].append(ask_reason[i])

                    if self._should_ask(ended[i], q_t_list[i]):
                        traj[i]['subgoals'].append(
                            ' '.join([self.nav_actions[a] for a in subgoals[i]]))

                    if a_t_list[i] == self.nav_actions.index('<end>') or \
                       time_step >= ob['traj_len'] - 1:
                        ended[i] = True

                assert queries_unused[i] >= 0

            # Early exit if all ended
            if ended.all():
                break

        if not self.is_eval:
            self._compute_loss(iter_idx)

        return traj

    def _compute_loss(self, iter_idx=None):

        self.loss = self.nav_loss + self.ask_loss
        iter_loss_avg = self.loss.item() / self.episode_len
        self.losses.append(iter_loss_avg)
        # if iter_idx: self.SW.add_scalar('train loss per iter', iter_loss_avg, iter_idx)

        iter_nav_loss_avg = self.nav_loss.item() / self.episode_len
        self.nav_losses.append(iter_nav_loss_avg)
        # if iter_idx: self.SW.add_scalar('train nav loss per iter', iter_nav_loss_avg, iter_idx)

        if self.random_ask or self.ask_first or self.teacher_ask or self.no_ask:
            iter_ask_loss_avg = 0
        else:
            iter_ask_loss_avg = self.ask_loss.item() / self.episode_len
        self.ask_losses.append(iter_ask_loss_avg)
        # if iter_idx: self.SW.add_scalar('train nav loss per iter', iter_ask_loss_avg, iter_idx)

    def _setup(self, env, feedback):
        self.nav_feedback = feedback['nav']
        self.ask_feedback = feedback['ask']

        assert self.nav_feedback in self.feedback_options
        assert self.ask_feedback in self.feedback_options

        self.env = env
        self.teacher.add_scans(env.scans)
        self.advisor.add_scans(env.scans)
        self.losses = []
        self.nav_losses = []
        self.ask_losses = []

    def test(self, env, feedback, use_dropout=False, allow_cheat=False):
        ''' Evaluate once on each instruction in the current environment '''

        self.allow_cheat = allow_cheat
        self.is_eval = not allow_cheat
        self._setup(env, feedback)
        if use_dropout:
            self.model.train()
        else:
            self.model.eval()
        return BaseAgent.test(self, env)

    def train(self, env, optimizer, n_iters, feedback, idx):
        ''' Train for a given number of iterations '''

        self.is_eval = False
        self._setup(env, feedback)
        self.model.train()

         # timming bookkeeping
        time_keep = {
            'total_traj_time': 0.0,
            'initial_setup': 0.0,
            'mask_sampling': 0.0,
            'initial_setup_pertimestep': 0.0,
            'decode_tentative': 0.0,
            'logging_1': 0.0,
            'logging_2': 0.0,
            'logging_3': 0.0,
            'pop_state_to_obs_1': 0.0,
            'ask_teacher_for_next_action': 0.0,
            'ask_loss': 0.0,
            'compute_q_t': 0.0,
            'determine_final_ask': 0.0,
            'decode_final': 0.0,
            'pop_state_to_obs_2': 0.0,
            'ask_for_next_nav_action': 0.0,
            'nav_loss': 0.0,
            'adjust_a_t_and_write_env_action': 0.0,
            'adjust_a_t': 0.0,
            'voting_sampling': 0.0,
            'synchronization_before_stack': 0.0,
            'prepare_for_LSTM_decoder_next_timestep': 0.0,
            'prepare_for_meta_algo_next_timestep': 0.0,
            'get_ask_target_and_reason_for_logging': 0.0,
            'prepend_instruction_back_to_env': 0.0,
            'loop_through_batch_to_write_env_action': 0.0,
            'env_step': 0.0,
            'save_traj_output': 0.0,
            'compute_loss_per_batch': 0.0,
            'loss_backward_time': 0.0,
        }       

        last_traj = []
        for iter in range(1, n_iters + 1):
            optimizer.zero_grad()
            traj, iter_time_keep = self.rollout(idx + iter)
            if n_iters - iter <= 10:
                last_traj.extend(traj)

            start_time = time.time() 
            self.loss.backward()

            # existing
            optimizer.step()
            time_keep['loss_backward_time'] += time.time() - start_time

            # time keep for different processes in rollout
            for key in time_keep.keys():
                if key in iter_time_keep:
                    time_keep[key] += iter_time_keep[key]

        return last_traj, time_keep















