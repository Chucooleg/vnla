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


class NavigationAgent(BaseAgent):
    """
    Handle hparams and constraints that are applied to all agents
    Interface with the Environment -- get batch data, image features
    """

    nav_actions = ['left', 'right', 'up', 'down',
                   'forward', '<end>', '<start>', '<ignore>']
    env_actions = [
        (0,-1, 0), # left   30 deg
        (0, 1, 0), # right  30 deg
        (0, 0, 1), # up     30 deg
        (0, 0,-1), # down   30 deg
        (1, 0, 0), # forward to navigable location at ix 1
        (0, 0, 0), # <end>
        (0, 0, 0), # <start>
        (0, 0, 0)  # <ignore>
    ]
    
    # feedback controls how agent carry on in train/test environments
    feedback_options = ['teacher', 'argmax', 'sample'] 

    # NOTE leave this to subclass ActionImitationAgent or ValueEstimationAgent
    # ask_actions = ['dont_ask', 'ask', '<start>', '<ignore>']

    def __init__(self, model, hparams, device):  
        # NOTE left out should make advisor for next level
        super(NavigationAgent, self).__init__(hparams)

        # traj params
        self.episode_len = hparams.max_episode_length

        # encoder-decoder params
        self.model = model
        self.max_input_length = hparams.max_input_length
        self.n_subgoal_steps  = hparams.n_subgoal_steps # used if asking
        self.coverage_size = hparams.coverage_size if hasattr(hparams, 'coverage_size') else None

        # bootstrap params
        self.bootstrap = hparams.bootstrap
        self.n_ensemble = hparams.n_ensemble
        self.bernoulli_probability = hparams.bernoulli_probability
        self.bootstrap_majority_vote = hparams.bootstrap_majority_vote
        if self.bootstrap:
            assert (self.bernoulli_probability > 0.0), "boostrap bernoulli prob must be greater than 0.0"

        # other training params
        self.device = device
        self.random = random
        self.random.seed(hparams.seed)
        self.random_state = np.random.RandomState(999)
 
    def make_instruction_batch(self, obs):
        ''' Make variables for a batch of input instructions. '''
        # convert to token IDs
        seq_tensor = np.array([self.env.encode(ob['instruction']) for ob in obs])
        # length of seq before padding idx pos
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        assert sum([length <= 0 for length in seq_lengths]) == 0
        # max allowable seq length
        max_length = max(seq_lengths)
        assert max_length <= self.max_input_length  # from hparam
        # convert seq inputs into torch tensors for encoder
        seq_tensor = torch.from_numpy(seq_tensor).long().to(self.device)[:, :max_length]
        seq_lengths = torch.from_numpy(seq_lengths).long().to(self.device)
        # shape (batch_size, max_length), 1 for padded token positions
        mask = (seq_tensor == padding_idx)

        return seq_tensor, mask, seq_lengths

    def get_feature_variable(self, obs):
        ''' Make a variable for a batch of precomputed image features. '''
        # img feature size is NOT a hparam. downloaded as-is
        feature_size = obs[0]['feature'].shape[0]
        features = np.empty((len(obs),feature_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :] = ob['feature']
        # convert features to torch tensor for decoder
        return torch.from_numpy(features).to(self.device)

    def argmax(self, logit):
        '''argmax from logits to determine next action'''
        # detach() excludes the tensor from gradient tracking
        return logit.max(1)[1].detach()

    def _sample(self, logit):
        '''sample from logits to determine next action'''
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

    def next_action(self, name, logit, target, feedback):
        ''' Determine the next action to take based on the training algorithm. '''
        if feedback == 'teacher':
            return target
        if feedback == 'argmax':
            return self.argmax(logit)
        if feedback == 'sample':
            return self._sample(logit)
        sys.exit('Invalid feedback option')
