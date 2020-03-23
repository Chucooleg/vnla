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

# sys.path.append('../../build/')
sys.path.append('/opt/MatterSim/build/')
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
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)

    def getStates(self):
        feature_states = []
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

        self.traj_len_estimates = defaultdict(list)

        self.query_ratio = hparams.query_ratio

        self.no_room = hasattr(hparams, 'no_room') and hparams.no_room

        if self.split is not None:
            print ("Using env split = {}".format(self.split))
            self.load_data(load_datasets(
                splits=[split], 
                path=hparams.data_path, 
                prefix='noroom' if self.no_room else 'asknav', 
                suffix=hparams.data_suffix if hasattr(hparams, 'data_suffix') else ''))

        # Estimate time budget using the upper 95% confidence bound
        if traj_len_estimates is None:
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
        self.data = []
        self.scans = set()
        for item in data:
            self.scans.add(item['scan'])
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
        if self.ix == 0:
            self.random.shuffle(self.data)
            print ("Shuffle, setting data ix at 0.")
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            self.random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        self.batch = batch

    def set_data_and_scans(self, data, scans):
        self.data = data
        self.scans = scans

    def reset_epoch(self):
        self.ix = 0

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
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
                'goal_viewpoints' : [path[-1] for path in item['paths']],
                'init_viewpoint' : item['paths'][0][0]
            })
            obs[-1]['max_queries'] = self.max_queries_constraints[i]
            obs[-1]['traj_len'] = self.traj_lens[i]
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
        return obs

    def _calculate_max_queries(self, traj_len):
        ''' Sample a help-requesting budget given a time budget. '''

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
            self.traj_lens[i] = min(self.max_episode_length, int(round(traj_len_estimate)))

            # Assign help-requesting budget
            self.max_queries_constraints[i] = self._calculate_max_queries(self.traj_lens[i])
            assert not math.isnan(self.max_queries_constraints[i])

        return self._get_obs()

    def step(self, actions):
        self.env.makeActions(actions)
        return self._get_obs()

    def prepend_instruction(self, idx, instr):
        ''' Prepend subgoal to end-goal. '''

        self.instructions[idx] = instr + ' . ' + self.batch[idx]['instruction']

    def get_obs(self):
        return self._get_obs()


class VNLABuildPretrainBatch():

    def __init__(self, hparams, split=None, from_train_env=None):
        self.env = EnvBatch(
            from_train_env=from_train_env.env if from_train_env is not None else None,
            img_features=hparams.img_features, batch_size=hparams.batch_size)

        self.split = split
        self.batch_size = hparams.batch_size

        if self.split is not None:
            print ("Using env split = {}".format(self.split))
            self._load_raw_data(load_datasets(
                splits=[split], 
                path=hparams.data_path, 
                prefix='asknav', 
                suffix=hparams.data_suffix if hasattr(hparams, 'data_suffix') else ''))

    def reset_epoch(self):
        self.ix = 0   

    def _load_raw_data(self, data):
        # 1. Load entire training dataset.
        self.raw_data = []
        self.raw_traj_lens = []
        self.scans = set()
        for item in data:
            self.scans.add(item['scan'])
            for i in range(len(item['trajectories'])):
                self.raw_traj_lens.append(len(item['trajectories'][i]))
                new_item = dict(item)
                # one instruction can have more than one gold trajectory
                new_item['instr_id'] = '%s_%d' % (item['path_id'], i)
                new_item['instruction'] = new_item['instructions'][0]
                del new_item['instructions']
                new_item['trajectory'] = new_item['trajectories'][i]
                del new_item['trajectories']
                self.raw_data.append(new_item)
        # 2. Sort training dataset by trajectory length.
        self.sorted_indices = np.argsort(self.raw_traj_lens)

        self.reset_epoch()

        if self.split is not None:
            print('VNLABuildPretrainBatch loaded with %d instructions, using split: %s' % (
                len(self.raw_data), self.split))

    def _next_raw_minibatch(self):
        '''
        Return a minibatch to build pretraining lookup. No shuffling.
        '''
        batch_indices = self.sorted_indices[self.ix:self.ix+self.batch_size]
        if len(batch_indices) < self.batch_size:
            self.ix = self.batch_size - len(batch_indices)
            batch_indices = np.append(batch_indices, self.sorted_indices[:self.ix])
        else:
            self.ix += self.batch_size
        self.batch = [self.raw_data[idx] for idx in batch_indices]
        self.traj_lens = [self.raw_traj_lens[idx] for idx in batch_indices]

    def _get_obs(self):
        obs = []
        gold_trajs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
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
                'goal_viewpoints' : [path[-1] for path in item['paths']],
                'init_viewpoint' : item['paths'][0][0],
            })
            obs[-1]['traj_len'] = self.traj_lens[i]
            gold_trajs.append(item['trajectory'])  # append a list of sim level actions

        return obs, gold_trajs

    def reset(self):
        ''' Load a new minibatch / episodes. '''
        self._next_raw_minibatch()

        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['paths'][0][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.instructions = [item['instruction'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        self.env.makeActions(actions)
        return self._get_obs()

    def get_obs(self):
        return self._get_obs()


class VNLAPretrainBatch():

    def __init__(self, hparams, split=None, from_train_env=None):
        self.env = EnvBatch(
            from_train_env=from_train_env.env if from_train_env is not None else None,
            img_features=hparams.img_features, batch_size=hparams.batch_size)

        self.random = random
        self.random.seed(hparams.seed)

        self.batch_size = hparams.batch_size
        self.window_size = hparams.window_size
        self.split = split
        self.train_mode = self.split not in ('val_seen', 'val_unseen')

        self.data_path = os.path.join(hparams.data_path, 'asknav_pretrain_lookup.json')
        if split is not None:
            self.data_path = self.data_path.replace('.json', '_{}.json'.format(split))

        self.ix = 0

    def _load_data(self):
        # Load pretrain train and val datasets into memory
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)

    def reset_epoch():
        self.ix == 0

    def _sample_swap_pos(d):
        # pick a position k within window to swap
        # image feature for position k will be swapped with that of k+1
        traj_len = len(d['trajectory'])
        swap_pos = np.random.choice(min(traj_len-1, 8-1))
        return swap_pos

    def _sample_window_pos(d):
        # sample a window of size 8 or max traj length
        traj_len = len(d['trajectory'])
        if traj_len <= 8:
            return (0, traj_len-1)
        else:
            start_pos = np.random.choice(traj_len-8+1)
            end_pos = start_pos + 8 - 1
            return (start_pos, end_pos)
        
    def _sample_frame(batch_data):
        # sample windows and swap pos for all datapoints
        for (i,d) in enumerate(batch_data):
            batch_data[i]['win_start_pos'], batch_data[i]['win_end_pos'] = sample_window_pos(d)
            swap_bool = np.random.binomial(1, 0.5)
            if swap_bool:
                batch_data[i]['swap_pos'] = sample_swap_pos(d)
            else:
                batch_data[i]['swap_pos'] = None
        return batch_data

    def _next_minibatch(self):
        '''grab next minibatch, sample window and swap pos if training'''
        if self.ix = 0:
            self.random.shuffle(self.data)
            print ("Shuffle, setting data ix at 0.")
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            self.random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size

        if self.train_mode:
            batch = self._sample_frame(batch)
        self.batch = batch

    def generate_next_minibatch(self):
        '''generate a window of env actions, a window of image features, gold swapped boolean, list of instr_ids'''

        a_t = np.empty((self.batch_size, 8, 3), dtype=np.float32)
        f_t = np.empty((self.batch_size, 8, 2048), dtype=np.float32)
        swapped_target = np.zeros(self.batch_size, dtype=np.float32)
        instr_ids = []

        for (i,d) in self.batch:

            instr_ids.append(d['instr_id'])

            # Extract window
            # a sequence of [(0,0,0), (0,0,1),...]
            env_action_window = d['trajectory'][d['win_start_pos']:d['win_end_pos']+1]
            # a sequence of [(vertex, viewix), (vertex, viewix),...]
            vertex_viewix_window = d['agent_path'][d['win_start_pos']:d['win_end_pos']+1]
            assert len(env_action_window) == len(vertex_viewix_window)

            # Swap (vertex, viewix) of pos k with k+1
            swap_pos = d['swap_pos']
            if swap_ops is not None:
                swapped_target[i] = 1
                vertex_viewix_pos_k = vertex_viewix_window[d['swap_pos']]
                vertex_viewix_window[d['swap_pos']] = vertex_viewix_window[d['swap_pos']+1]
                vertex_viewix_window[d['swap_pos']+1] = vertex_viewix_pos_k

            # Pad short windows back to 8
            ct = 0
            while len(env_action_window) < 8:
                if ct % 2:
                    env_action_window.insert(0, env_action_window[0])
                    vertex_viewix_window.insert(0, vertex_viewix_window[0])
                else:
                    env_action_window.insert(-1, env_action_window[-1])
                     vertex_viewix_window.insert(-1, vertex_viewix_window[-1])
                ct += 1
            assert len(env_action_window) == len(vertex_viewix_window) == 8
            a_t[i] = np.array(env_action_window)

            # Lookup the image feature vector
            for j in range(8):
                long_id = d['scan'] + '_' + vertex_viewix_window[j][0]
                f_t[i][j] = self.env.features[long_id][vertex_viewix_window[j][1], :]
            
        return a_t, f_t, swapped_target, instr_ids