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
from oracle import make_oracle


class ActionImitationAgent(NavigationAgent):
    """
    Handle hparams and constraints that are applied to action imitation agents, w/o asking.
    Train and test are implemented here in case training and testing are different in value estimation agents with the use of history buffer.
    """
    ask_actions = ['dont_ask', 'ask', '<start>', '<ignore>']

    # NOTE override attributes here if necessary
    # nav_actions = NavigationAgent.nav_actions
    # env_actions = NavigationAgent.env_actions

    def __init__(self, model, hparams, device):
        super(ActionImitationAgent, self).__init__(model, hparams, device)

        self.nav_criterion = nn.CrossEntropyLoss(
            ignore_index=self.nav_actions.index('<ignore>'))
        
        self.nav_teacher = make_oracle('shortest', self.nav_actions, self.env_actions)

        # accumulated in subclasses
        self.loss = 0.0   
        # only for computing validation losses using training conditions
        self.allow_cheat = False  
        # compute losses if not eval (i.e. training)
        self.is_eval = False  

    def _setup(self, env, feedback):
        # Leave for subclasses to:
        # - setup nav, ask feedback
        # - setup nav, ask losses
        # - setup self.env
        # - add scans to teachers/advisors
        raise NotImplementedError('Subclasses are expected to implement _setup')      

    def test(self, env, feedback, use_dropout=False, allow_cheat=False):
        ''' Evaluate once on each instruction in the current environment '''

        self.allow_cheat = allow_cheat
        self.is_eval = not allow_cheat
        self._setup(env, feedback)
        if use_dropout:
            self.model.train()
        else:
            self.model.eval()
        return self.base_test()

    def train(self, env, optimizer, n_iters, feedback, idx):
        '''Train for a given number `n_iters` of iterations. 
        `n_iters` is hparams.log_every (default 1000) or remaining.
        '''

        self.is_eval = False

        # Set up self.env, feedback(teacher/argmax/learned)
        # Initialize losses, add scans to teachers/advisors
        self._setup(env, feedback)

        # do not use dropout
        self.model.train()

        # time report
        time_report = defaultdict(int)

        # List of 10*batch_size number of trajectory dictionaries
        last_traj = []

        # Loop through 1000 batches of datapts
        for itr in range(1, n_iters+1):

            # Global training iteration index
            global_iter_idx = idx + itr

            # Rollout the agent, collect losses
            traj, iter_time_report = self.rollout(global_iter_idx)

            # Keep time for rollout processes
            for key in iter_time_report.keys():
                time_report[key] += iter_time_report[key]

            # Save rollout trajectories from last 10 batches to compute sample stats (e.g. compute_ask_stats)
            if n_iters - itr <= 10:
                last_traj.extend(traj)

           # Compute gradients
            start_time = time.time()
            optimizer.zero_grad()
            self.loss.backward()

            # NOTE add code to divide up gradients by 1/n_ensemble in the shared encoder if we are bootstrapping
            # NOTE add code here if gradient clipping

            # Update torch model params
            optimizer.step()
            time_report['backprop_time'] += time.time() - start_time
        
        return last_traj, time_report


