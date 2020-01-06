from __future__ import division
import os
import sys
import csv
import numpy as np
import math
import base64
import json
import random
import networkx as nx
from collections import defaultdict
import scipy.stats

# sys.path.append('../../build/')  # if local
sys.path.append('/opt/MatterSim/build/')  # if docker or Philly
import MatterSim

from oracle import make_oracle
from utils import load_datasets, load_nav_graphs
import utils

csv.field_size_limit(sys.maxsize)


class EnvBatch():

    def __init__(self, from_train_env=None, img_features=None, batch_size=100):
        if from_train_env is not None:
            self.features = from_train_env.features
            self.image_h  = from_train_env.image_h
            self.image_w  = from_train_env.image_w
            self.vfov     = from_train_env.vfov
        elif img_features is not None:
            self.image_h, self.image_w, self.vfov, self.features = \
                utils.load_img_features(img_features)
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.sims = []
        # assign a new simulator for each data point in the batch
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.setNavGraphPath(
                os.path.join(os.getenv('PT_DATA_DIR'), 'connectivity'))
            sim.init()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        '''set x_curr and psi_curr for each simulator in beginning of episodes'''
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)

    def getStates(self):
        '''Get image feature and general states (graph / simulator env) info from current batch of simulators

        - feature refers to image feature
        - state has attributes .scanId .location.point .location.viewpointId, .viewIndex .heading .elevation .step .navigableLocations
        '''
        feature_states = []
        # self.sims has length batch_size
        for sim in self.sims:
            state = sim.getState()
            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id][state.viewIndex,:]
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions):
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)


class VNLAExplorationBatch():

    def __init__(self, batch_size=100):
        # assign a new simulator for each data point in the batch
        self.batch_size = batch_size
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.setNavGraphPath(
                os.path.join(os.getenv('PT_DATA_DIR'), 'connectivity'))
            sim.init()
            self.sims.append(sim)

    def reset(self, obs):
        '''set x_curr and psi_curr for each simulator in beginning of episodes'''
        for i, ob in enumerate(obs):
            self.sims[i].newEpisode(ob['scan'], ob['viewpoint'], ob['heading'], ob['elevation'])

    def _take_explore_step(self, instruction):
        '''
        1. Take 1 step/macro-step and collect the viewIndex
        2. Check if there's a reachable vertex within 30 deg, return the closest one if so.

        instruction : list len = batch_size, each element is a tuple of single or macro-action.

        Returns: 
            A list of tuples. list len = batch_size.
            Each element in list is a tuple (view_ix, list of env action tuples, vertex string)
        '''
        # TODO solve how oracle provides instructions first
        [(view_ix, efficient_action_list, vertex_str), ...]
        # assert this thing is batch size first

    def _explore_horizontally(self, viewix_env_actions_map, viewix_next_vertex_map, heading_adjusts):
        '''turn around horizontally - 11 times'''
        for _ in range(11):
            # list, length = batch_size, [(view_ix, efficient actions, vertex), ...]
            viewix_actions_vertex_tuples = self._take_explore_step(heading_adjusts)
            # loop through batch
            for i, (viewix, env_actions_list, vertex_str) in enumerate(viewix_actions_vertex_tuples):
                viewix_env_actions_map[i][viewix] = env_actions_list
                viewix_next_vertex_map[i][viewix] = vertex_str
        return viewix_env_actions_map, viewix_next_vertex_map


    def explore(self, instructions):
        '''
        instructions: 3 lists, each of len batch_size.

        Returns:
            viewix_env_actions_map: list (batch_size, 36, varies). Each [(0,1,0), (0,-1,1), ...]
            viewix_next_vertex_map: list (batch_size, 36, varies). Each string.
        ''' 
        viewix_env_actions_map = [[None] * 36] * self.batch_size
        viewix_next_vertex_map = [[''] * 36] * self.batch_size

        # elaborate condensed instructions into 35 steps that cover the sphere
        heading_adjusts, elevation_adjusts_1, elevation_adjusts_2 = instructions

        # capture current - 1 time
        # TODO

        # turn around horizontally - 11 times
        viewix_env_actions_map, viewix_next_vertex_map = \
            self._explore_horizontally(viewix_env_actions_map, viewix_next_vertex_map, heading_adjusts)

        # adjust elevation - 1 time
        # TODO

        # turn around horizontally - 11 times
        viewix_env_actions_map, viewix_next_vertex_map = \
            self._explore_horizontally(viewix_env_actions_map, viewix_next_vertex_map, heading_adjusts)

        # adjust elevation - 1 time
        # TODO

        # turn around horizontally - 11 times
        viewix_env_actions_map, viewix_next_vertex_map = \
            self._explore_horizontally(viewix_env_actions_map, viewix_next_vertex_map, heading_adjusts)

        # assert all 36 views are covered

        return viewix_env_actions_map, viewix_next_vertex_map

    def _getStates(self):
        pass

    def _get_obs(self):
        # '''get obs for current batch from self.data and simulators'''
        # obs = []
        # # feature, state are from the current batch of simulators
        # for i, (feature, state) in enumerate(self.env.getStates()):
        #     # item is from the current batch of data loaded from self.data
        #     item = self.batch[i]
        #     obs.append({
        #         'instr_id' : item['instr_id'],
        #         'scan' : state.scanId,
        #         'point': state.location.point,
        #         'viewpoint' : state.location.viewpointId,
        #         'viewIndex' : state.viewIndex,
        #         'heading' : state.heading,
        #         'elevation' : state.elevation,
        #         'feature' : feature,
        #         'step' : state.step,
        #         'navigableLocations' : state.navigableLocations,
        #         'instruction' : self.instructions[i],
        #         # there can be more than 1 goal viewpoints for each datapt
        #         'goal_viewpoints' : [path[-1] for path in item['paths']],
        #         'init_viewpoint' : item['paths'][0][0]
        #     })
        #     # k computed when we load a new mini batch
        #     obs[-1]['max_queries'] = self.max_queries_constraints[i]
        #     # ^T computed when we load a new mini batch
        #     obs[-1]['traj_len'] = self.traj_lens[i]
        #     # # not sure if this is used anywhere
        #     # if 'instr_encoding' in item:
        #     #     obs[-1]['instr_encoding'] = item['instr_encoding']
        # return obs


class VNLABatch():

    def __init__(self, hparams, split=None, tokenizer=None, from_train_env=None,
                 traj_len_estimates=None):
        self.env = EnvBatch(
            from_train_env=from_train_env.env if from_train_env is not None else None,
            img_features=hparams.img_features, batch_size=hparams.batch_size)

        self.random = random
        self.random.seed(hparams.seed)

        self.tokenizer = tokenizer
        self.split = split
        self.batch_size = hparams.batch_size
        self.max_episode_length = hparams.max_episode_length
        self.n_subgoal_steps = hparams.n_subgoal_steps

        # time budget ^T
        self.traj_len_estimates = defaultdict(list)
        # ratio used to compute k
        self.query_ratio = hparams.query_ratio

        # we are not experimenting with NoRoom dataset
        self.no_room = hasattr(hparams, 'no_room') and hparams.no_room

        if self.split is not None:
            self.load_data(load_datasets([split], hparams.data_path,
                prefix='noroom' if self.no_room else 'asknav'))

        # Estimate time budget ^T
        # key k -- (item['start_region_name'], item['end_region_name'])
        if traj_len_estimates is None:
            # training - estimate time budget using the upper 95% confidence bound
            for k in self.traj_len_estimates:
                self.traj_len_estimates[k] = min(self.max_episode_length,
                    float(np.average(self.traj_len_estimates[k]) +
                    1.95 * scipy.stats.sem(self.traj_len_estimates[k])))
                assert not math.isnan(self.traj_len_estimates[k])
        else:
            for k in self.traj_len_estimates:
                if k in traj_len_estimates:
                    self.traj_len_estimates[k] = traj_len_estimates[k]
                else:
                    self.traj_len_estimates[k] = self.max_episode_length

    def make_traj_estimate_key(self, item):
        if self.no_room:
            key = (item['start_region_name'], item['object_name'])
        else:
            key = (item['start_region_name'], item['end_region_name'])
        return key

    def encode(self, instr):
        if self.tokenizer is None:
            sys.exit('No tokenizer!')
        return self.tokenizer.encode_sentence(instr)

    def load_data(self, data):
        '''
        Load entire dataset, not just a batch
        Load the instructions, instr_id to self.data
        Load the scans to self.scans
        '''
        self.data = []
        self.scans = set()
        for item in data:
            self.scans.add(item['scan'])
            # key -- (item['start_region_name'], item['end_region_name'])
            key = self.make_traj_estimate_key(item)
            self.traj_len_estimates[key].extend(
                len(t) for t in item['trajectories'])

            for j,instr in enumerate(item['instructions']):
                new_item = dict(item)
                del new_item['instructions']
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instruction'] = instr
                self.data.append(new_item)

        self.reset_epoch()

        if self.split is not None:
            print('VNLABatch loaded with %d instructions, using split: %s' % (
                len(self.data), self.split))

    def _next_minibatch(self):
        '''put the next batch of data into self.batch from self.data'''
        if self.ix == 0:
            self.random.shuffle(self.data)
        batch = self.data[self.ix:self.ix+self.batch_size]
        # if the batch runs out of data pt
        if len(batch) < self.batch_size:
            self.random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        self.batch = batch

    def reset_epoch(self):
        '''start a new epoch. 
        called in train.test() and load_data() '''
        self.ix = 0

    def _get_obs(self):
        '''get obs for current batch from self.data and simulators'''
        obs = []
        # feature, state are from the current batch of simulators
        for i, (feature, state) in enumerate(self.env.getStates()):
            # item is from the current batch of data loaded from self.data
            item = self.batch[i]
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'point': state.location.point,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'step' : state.step,
                'navigableLocations' : state.navigableLocations,
                'instruction' : self.instructions[i],
                # there can be more than 1 goal viewpoints for each datapt
                'goal_viewpoints' : [path[-1] for path in item['paths']],
                'init_viewpoint' : item['paths'][0][0]
            })
            # k computed when we load a new mini batch
            obs[-1]['max_queries'] = self.max_queries_constraints[i]
            # ^T computed when we load a new mini batch
            obs[-1]['traj_len'] = self.traj_lens[i]
            # # not sure if this is used anywhere
            # if 'instr_encoding' in item:
            #     obs[-1]['instr_encoding'] = item['instr_encoding']
        return obs

    def _calculate_max_queries(self, traj_len):
        ''' Sample a help-requesting budget k given a time budget ^T. '''

        max_queries = self.query_ratio * traj_len / self.n_subgoal_steps
        int_max_queries = int(max_queries)
        frac_max_queries = max_queries - int_max_queries
        return int_max_queries + (self.random.random() < frac_max_queries)

    def reset(self, is_eval):
        ''' Load a new minibatch / episodes. '''

        self._next_minibatch()

        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['paths'][0][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.instructions = [item['instruction'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)

        self.max_queries_constraints = [None] * self.batch_size
        self.traj_lens = [None] * self.batch_size

        for i, item in enumerate(self.batch):
            # Assign time budget
            if is_eval:
                # If eval use expected trajectory length between start_region and end_region
                key = self.make_traj_estimate_key(item)
                traj_len_estimate = self.traj_len_estimates[key]
            else:
                # If train use average oracle trajectory length
                traj_len_estimate = sum(len(t)
                    for t in item['trajectories']) / len(item['trajectories'])
            # ^T
            self.traj_lens[i] = min(self.max_episode_length, int(round(traj_len_estimate)))

            # Assign help-requesting budget k
            self.max_queries_constraints[i] = self._calculate_max_queries(self.traj_lens[i])
            assert not math.isnan(self.max_queries_constraints[i])

        return self._get_obs()

    def step(self, actions):
        self.env.makeActions(actions)
        return self._get_obs()

    def steps(self, multiple_actions):
        '''
        multiple_actions : list of len=batch_size
                           each element is a list of env action tuples
        '''
        assert len(self.sims) == len(multiple_actions)
        for i in range(len(self.sims)):
            # a list of tuples
            actions = multiple_actions[i]
            for index, heading, elevation in actions:
                self.sims[i].makeAction(index, heading, elevation)
        return self._get_obs()

    def prepend_instruction(self, idx, instr):
        ''' Prepend subgoal to end-goal. '''

        self.instructions[idx] = instr + ' . ' + self.batch[idx]['instruction']

    def get_obs(self):
        return self._get_obs()



