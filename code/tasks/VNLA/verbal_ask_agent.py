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

    def __init__(self, model, hparams, room_types, device):
        super(VerbalAskAgent, self).__init__(model, hparams, room_types, device,
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

    def prepend_instruction_to_obs(self, initial_instr, idx, instr):
        return instr + ' . ' + initial_instr[idx]

    def pack_heads_with_diff_seq_lens(self, list_of_heads, dattype=torch.float):
        """list of heads : list len=n_ensemble; first dim of each element=batch_size, 
        second dim of each element is the max seq lengths corresponding to each head. 
        These second dim varies between heads and we need to equalize them before stacking."""
        max_seq_len_among_heads = max([h.shape[1] for h in list_of_heads])
        for k in range(len(list_of_heads)):
            if list_of_heads[k].shape[1] != max_seq_len_among_heads:
                short = list_of_heads[k].shape[1]
                new_shape = list(list_of_heads[k].shape)
                new_shape[1] = max_seq_len_among_heads
                # TODO constant
                replacement_tensor = torch.zeros(new_shape, dtype=dattype, device=self.device)
                replacement_tensor[:, :short] = list_of_heads[k]
                list_of_heads[k] = replacement_tensor
        return [h.type(dattype) for h in list_of_heads]

    def rollout(self, iter_idx=None):
        if self.swap_images:
            return self._rollout_with_image_swapping(iter_idx)
        else:
            return self._rollout(iter_idx)

    def _rollout_with_image_swapping(self, iter_idx=None):

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
        }

        traj_start_time = time.time()
        start_time = time.time() 

        # Reset environment
        obs = self.env.reset(self.allow_cheat)
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
        
        # Initialize image features
        f_t = torch.zeros(batch_size, self.img_feature_size, dtype=torch.float, device=self.device)

        # Whether agent decides to stop
        ended = np.array([False] * batch_size)

        self.nav_loss = 0
        self.ask_loss = 0
        self.room_loss = 0

        action_subgoals = [[] for _ in range(batch_size)]
        n_subgoal_steps = [0] * batch_size

        env_action = [None] * batch_size
        queries_unused = [ob['max_queries'] for ob in obs]

        episode_len = max(ob['traj_len'] for ob in obs)

        time_keep['initial_setup'] += time.time() - start_time 

        for time_step in range(episode_len):

            start_time = time.time() 

            # Mask out invalid actions
            nav_logit_mask = torch.zeros(batch_size,
                AskAgent.n_output_nav_actions(), dtype=torch.bool, device=self.device)
            ask_logit_mask = torch.zeros(batch_size,
                AskAgent.n_output_ask_actions(), dtype=torch.bool, device=self.device)

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
            if self.allow_cheat:
                swap_image = np.random.binomial(1, self.gamma if self.swap_first else (1 - self.gamma))
            else:
                swap_image = 0
            if swap_image:
                f_t = self._feature_variable_random(obs)
            else:
                f_t = self._feature_variable(obs)

            # Budget features
            b_t = torch.tensor(queries_unused, dtype=torch.long, device=self.device)

            time_keep['initial_setup_pertimestep'] += time.time() - start_time

            # Run first forward pass to compute ask logit
            start_time = time.time() 
            _, _, nav_logit, nav_softmax, ask_logit, ask_softmax, _ = self.model.decode(
                a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask,
                ask_logit_mask, budget=b_t, cov=cov)
            time_keep['decode_tentative'] += time.time() - start_time

            # logging only
            start_time = time.time() 
            nav_logits_initial_list = nav_logit.data.tolist()
            nav_softmax_initial_list = nav_softmax.data.tolist()
            ask_logits_list = ask_logit.data.tolist()
            ask_softmax_list = ask_softmax.data.tolist()
            nav_logit_masks_list = nav_logit_mask.data.tolist()
            ask_logit_masks_list = ask_logit_mask.data.tolist()
            time_keep['logging_1'] += time.time() - start_time

            start_time = time.time()
            self._populate_agent_state_to_obs(obs, nav_softmax, queries_unused, traj, ended, time_step)
            time_keep['pop_state_to_obs_1'] += time.time() - start_time

            # Ask teacher for next ask action
            start_time = time.time()
            ask_target, ask_reason = self.teacher.next_ask(obs)
            ask_target = torch.tensor(ask_target, dtype=torch.long, device=self.device)
            time_keep['ask_teacher_for_next_action'] += time.time() - start_time

            start_time = time.time()
            if self.allow_cheat and not (self.random_ask or self.ask_first or self.teacher_ask or self.no_ask):
                self.ask_loss += self.ask_criterion(ask_logit, ask_target)
            time_keep['ask_loss'] += time.time() - start_time
                
            # Determine next ask action
            start_time = time.time()
            q_t = self._next_action('ask', ask_logit, ask_target, self.ask_feedback)
            time_keep['compute_q_t'] += time.time() - start_time

            start_time = time.time()
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
            time_keep['determine_final_ask'] += time.time() - start_time

            start_time = time.time()
            # Run second forward pass to compute nav logit
            # NOTE: q_t and b_t changed since the first forward pass.
            q_t = torch.tensor(q_t_list, dtype=torch.long, device=self.device)
            # b_t = torch.tensor(queries_unused, dtype=torch.long, device=self.device)
            decoder_h, _, nav_logit, nav_softmax, cov = self.model.decode_nav(
                a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask, cov=cov)
            time_keep['decode_final'] += time.time() - start_time

            # logging only
            start_time = time.time()
            nav_logits_final_list = nav_logit.data.tolist()
            nav_softmax_final_list = nav_softmax.data.tolist()
            time_keep['logging_2'] += time.time() - start_time

            # Repopulate agent state
            # NOTE: queries_unused may have changed but it's fine since nav_teacher does not use it!
            start_time = time.time()
            self._populate_agent_state_to_obs(obs, nav_softmax, queries_unused,
                traj, ended, time_step)
            time_keep['pop_state_to_obs_2'] += time.time() - start_time

            # Ask teacher for next nav action
            start_time = time.time()
            nav_target = self.teacher.next_nav(obs)
            nav_target = torch.tensor(nav_target, dtype=torch.long, device=self.device)
            time_keep['ask_for_next_nav_action'] += time.time() - start_time

            # logging only
            start_time = time.time()
            nav_target_list = nav_target.data.tolist()
            time_keep['logging_3'] += time.time() - start_time

            # Nav loss
            start_time = time.time()
            if self.allow_cheat:
                self.nav_loss += self.nav_criterion(nav_logit, nav_target)
            time_keep['nav_loss'] += time.time() - start_time

            # Determine next nav action
            start_time = time.time()
            a_t = self._next_action('nav', nav_logit, nav_target, self.nav_feedback)

            # Translate agent action to environment action
            a_t_list = a_t.data.tolist()
            for i in range(batch_size):
                # Conditioned on teacher action during intervention
                # (training only or when teacher_interpret flag is on)
                if (self.teacher_interpret or self.allow_cheat) and \
                        n_subgoal_steps[i] < len(action_subgoals[i]):
                    a_t_list[i] = action_subgoals[i][n_subgoal_steps[i]]
                    n_subgoal_steps[i] += 1

                env_action[i] = self.teacher.interpret_agent_action(a_t_list[i], obs[i])
            
            a_t = torch.tensor(a_t_list, dtype=torch.long, device=self.device)
            time_keep['adjust_a_t_and_write_env_action'] += time.time() - start_time

            # Take nav action
            start_time = time.time()
            obs = self.env.step(env_action)
            time_keep['env_step'] += time.time() - start_time

            # import pdb; pdb.set_trace()

            # Save trajectory output
            start_time = time.time()
            ask_target_list = ask_target.data.tolist()
            time_keep['get_ask_target_and_reason_for_logging'] += time.time() - start_time

            start_time = time.time()
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
            time_keep['save_traj_output'] += time.time() - start_time

        # # check GPU usage
        # import pdb; pdb.set_trace()

        if self.allow_cheat:
            start_time = time.time()
            self._compute_loss(iter_idx)
            time_keep['compute_loss_per_batch'] += time.time() - start_time

        time_keep['total_traj_time'] += time.time() - traj_start_time
        return traj, time_keep

    def _rollout(self, iter_idx=None):

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
        }

        traj_start_time = time.time()
        start_time = time.time() 

        # Reset environment
        obs = self.env.reset(self.allow_cheat)
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
        
        # Initialize image features
        f_t = torch.zeros(batch_size, self.img_feature_size, dtype=torch.float, device=self.device)

        # Whether agent decides to stop
        ended = np.array([False] * batch_size)

        self.nav_loss = 0
        self.ask_loss = 0
        self.room_loss = 0

        action_subgoals = [[] for _ in range(batch_size)]
        n_subgoal_steps = [0] * batch_size

        env_action = [None] * batch_size
        queries_unused = [ob['max_queries'] for ob in obs]

        episode_len = max(ob['traj_len'] for ob in obs)

        time_keep['initial_setup'] += time.time() - start_time 

        for time_step in range(episode_len):

            start_time = time.time() 

            # Mask out invalid actions
            nav_logit_mask = torch.zeros(batch_size,
                AskAgent.n_output_nav_actions(), dtype=torch.bool, device=self.device)
            ask_logit_mask = torch.zeros(batch_size,
                AskAgent.n_output_ask_actions(), dtype=torch.bool, device=self.device)

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

            time_keep['initial_setup_pertimestep'] += time.time() - start_time

            # Run first forward pass to compute ask logit
            start_time = time.time() 
            _, _, nav_logit, nav_softmax, ask_logit, ask_softmax, _ = self.model.decode(
                a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask,
                ask_logit_mask, budget=b_t, cov=cov)
            time_keep['decode_tentative'] += time.time() - start_time

            # logging only
            start_time = time.time() 
            nav_logits_initial_list = nav_logit.data.tolist()
            nav_softmax_initial_list = nav_softmax.data.tolist()
            ask_logits_list = ask_logit.data.tolist()
            ask_softmax_list = ask_softmax.data.tolist()
            nav_logit_masks_list = nav_logit_mask.data.tolist()
            ask_logit_masks_list = ask_logit_mask.data.tolist()
            time_keep['logging_1'] += time.time() - start_time

            start_time = time.time()
            self._populate_agent_state_to_obs(obs, nav_softmax, queries_unused, traj, ended, time_step)
            time_keep['pop_state_to_obs_1'] += time.time() - start_time

            # Ask teacher for next ask action
            start_time = time.time()
            ask_target, ask_reason = self.teacher.next_ask(obs)
            ask_target = torch.tensor(ask_target, dtype=torch.long, device=self.device)
            time_keep['ask_teacher_for_next_action'] += time.time() - start_time

            start_time = time.time()
            if self.allow_cheat and not (self.random_ask or self.ask_first or self.teacher_ask or self.no_ask):
                self.ask_loss += self.ask_criterion(ask_logit, ask_target)
            time_keep['ask_loss'] += time.time() - start_time
                
            # Determine next ask action
            start_time = time.time()
            q_t = self._next_action('ask', ask_logit, ask_target, self.ask_feedback)
            time_keep['compute_q_t'] += time.time() - start_time

            start_time = time.time()
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
            time_keep['determine_final_ask'] += time.time() - start_time

            start_time = time.time()
            # Run second forward pass to compute nav logit
            # NOTE: q_t and b_t changed since the first forward pass.
            q_t = torch.tensor(q_t_list, dtype=torch.long, device=self.device)
            # b_t = torch.tensor(queries_unused, dtype=torch.long, device=self.device)
            decoder_h, _, nav_logit, nav_softmax, cov = self.model.decode_nav(
                a_t, q_t, f_t, decoder_h, ctx, seq_mask, nav_logit_mask, cov=cov)
            time_keep['decode_final'] += time.time() - start_time

            # logging only
            start_time = time.time()
            nav_logits_final_list = nav_logit.data.tolist()
            nav_softmax_final_list = nav_softmax.data.tolist()
            time_keep['logging_2'] += time.time() - start_time

            # Repopulate agent state
            # NOTE: queries_unused may have changed but it's fine since nav_teacher does not use it!
            start_time = time.time()
            self._populate_agent_state_to_obs(obs, nav_softmax, queries_unused,
                traj, ended, time_step)
            time_keep['pop_state_to_obs_2'] += time.time() - start_time

            # Ask teacher for next nav action
            start_time = time.time()
            nav_target = self.teacher.next_nav(obs)
            nav_target = torch.tensor(nav_target, dtype=torch.long, device=self.device)
            time_keep['ask_for_next_nav_action'] += time.time() - start_time

            # logging only
            start_time = time.time()
            nav_target_list = nav_target.data.tolist()
            time_keep['logging_3'] += time.time() - start_time

            # Nav loss
            start_time = time.time()
            if self.allow_cheat:
                self.nav_loss += self.nav_criterion(nav_logit, nav_target)
            time_keep['nav_loss'] += time.time() - start_time

            # Determine next nav action
            start_time = time.time()
            a_t = self._next_action('nav', nav_logit, nav_target, self.nav_feedback)

            # Translate agent action to environment action
            a_t_list = a_t.data.tolist()
            for i in range(batch_size):
                # Conditioned on teacher action during intervention
                # (training only or when teacher_interpret flag is on)
                if (self.teacher_interpret or self.allow_cheat) and \
                        n_subgoal_steps[i] < len(action_subgoals[i]):
                    a_t_list[i] = action_subgoals[i][n_subgoal_steps[i]]
                    n_subgoal_steps[i] += 1

                env_action[i] = self.teacher.interpret_agent_action(a_t_list[i], obs[i])
            
            a_t = torch.tensor(a_t_list, dtype=torch.long, device=self.device)
            time_keep['adjust_a_t_and_write_env_action'] += time.time() - start_time

            # Take nav action
            start_time = time.time()
            obs = self.env.step(env_action)
            time_keep['env_step'] += time.time() - start_time

            # import pdb; pdb.set_trace()

            # Save trajectory output
            start_time = time.time()
            ask_target_list = ask_target.data.tolist()
            time_keep['get_ask_target_and_reason_for_logging'] += time.time() - start_time

            start_time = time.time()
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
            time_keep['save_traj_output'] += time.time() - start_time

        # # check GPU usage
        # import pdb; pdb.set_trace()

        if self.allow_cheat:
            start_time = time.time()
            self._compute_loss(iter_idx)
            time_keep['compute_loss_per_batch'] += time.time() - start_time

        time_keep['total_traj_time'] += time.time() - traj_start_time
        return traj, time_keep