import json
import os
import sys
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.distributions as D
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from utils import padding_idx


class BaseAgent(object):
    """
    Handles test results writing
    """

    def __init__(self, hparams):
        random.seed(1)
        self.results = {}
        self.results_path = None
        self.SW = SummaryWriter(hparams.tensorboard_dir, flush_secs=30)

    def write_results(self):
        output = []
        for k, v in self.results.items():
            item = { 'instr_id' : k }
            item.update(v)
            output.append(item)

        with open(self.results_path, 'w') as f:
            try:
                json.dump(output, f)
            except:
                import ipdb; ipdb.set_trace()

    def rollout(self, iter_idx=None):
        raise NotImplementedError('Subclasses are expected to implement rollout')

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def base_test(self):
        # self.env is defined in subclass _setup()
        self.env.reset_epoch()

        # self.is_eval is set to True in subclass

        self.results = {}
        looped = False
        traj = []
        with torch.no_grad():
            # rollout many times until all the datapts are processed
            while True:
                for tr in self.rollout()[0]:
                    if tr['instr_id'] in self.results:
                        looped = True
                    else:
                        self.results[tr['instr_id']] = {
                                # common to all agents
                                'trajectory' : tr['agent_path'], 
                                'scan'       : tr['scan'],
                                'agent_nav'  : tr['agent_nav'],
                            }
                        for k in [
                            # imitation no ask agent
                            # single head
                            'agent_nav_logits', 'agent_nav_softmax', 
                            'agent_nav_logit_masks', 'teacher_nav',
                            # imitation verbal ask agent
                            # single head
                            'agent_nav_logits_tentative', 'agent_nav_softmax_tentative',
                            'agent_nav_logits_final', 'agent_nav_softmax_final',
                            'agent_ask',
                            'agent_ask_logits', 'agent_ask_softmax',
                            'agent_ask_logit_masks', 'teacher_ask', 'teacher_ask_reason',
                            # value estimation no ask no recovery agent
                            'agent_q_values', 'teacher_q_values', 'beta', 'expert_rollin_bool',
                            ]:

                            if k in tr:
                                self.results[tr['instr_id']][k] = tr[k]
                        if 'subgoals' in tr:
                            self.results[tr['instr_id']]['subgoals'] = tr['subgoals']
                        traj.append(tr)
                if looped:
                    break
        return traj

    def add_is_success(self, is_success):
        for k, v in is_success:
            self.results[k]['is_success'] = v


