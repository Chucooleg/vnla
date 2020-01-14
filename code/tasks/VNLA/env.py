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

from oracle import make_oracle
from utils import load_datasets, load_nav_graphs
import utils

try:
    sys.path.append('/opt/MatterSim/build/')  # local docker or Philly
    import MatterSim
except: 
    # local conda env only
    # sys.path.append('../../build/')  
    # print('cwd: %s' % os.getcwd())
    sys.path.append('/home/hoyeung/Documents/vnla/code/build')  
    import MatterSim

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

    def __init__(self, obs, agent_nav_actions, env_nav_actions):
        # assign a new simulator for each ob in the batch
        self.batch_size = len(obs)
        self.image_w = 640
        self.image_h = 480
        self.vfov = 60
        self.sims = []
        for i in range(self.batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.setNavGraphPath(
                os.path.join(os.getenv('PT_DATA_DIR'), 'connectivity'))
            sim.init()
            self.sims.append(sim)
        self._reset(obs)

        # list length=batch_size, each element is a list of 35 rotation tuples.
        # e.g. [[(0,1,0), (0,-1,1), ...], [(0,1,0), (0,-1,1), ...], ....]
        # [[None]] * self.batch_size
        self._rotation_history = [[] for _ in range(self.batch_size)]

        self.agent_nav_actions = agent_nav_actions
        self.env_nav_actions = env_nav_actions

    def _reset(self, obs):
        '''set x_curr and psi_curr for each simulator in beginning of episodes'''
        for i, ob in enumerate(obs):
            self.sims[i].newEpisode(ob['scan'], ob['viewpoint'], ob['heading'], ob['elevation'])

#-------------------

    def _getStates(self):
        '''
        Get the most current viewIndex and closest navigableLocation at index [1] of each sim.

        Returns:
            viewix_batch: list length = batch_size, each int.
            closest_vertex_batch: list length = batch_size, each a dictionary (see state.navigableLocations)
        '''
        viewix_batch = [None] * self.batch_size
        closest_vertex_batch = [None] * self.batch_size
        for (i, sim) in enumerate(self.sims):
            state = sim.getState()
            viewix_batch[i] = state.viewIndex
            if len(state.navigableLocations) > 1:
                closest_vertex_batch[i] = state.navigableLocations[1]
            else:
                closest_vertex_batch[i] = {}
        return viewix_batch, closest_vertex_batch

    def _map_to_efficient_rotation(self, rotation_history):
        '''
        Figure out shortest rotation sequence from rotation_history. 
        Returns:
            list. length = batch_size, 
            each element is a single list of tuples. [(0,1,0), (0,-1,1), ...]
        '''
        # env should have taken 1 or more rotations already
        assert all([len(hist) > 0 for hist in rotation_history])

        def compress_single(l):
            '''
            Return more compact and efficient list of env action tuples.
            Do not combine [(0,1,0), (0,0,1)] to [(0,1,1)] because they need to be translated into single action action indices for LSTM later.

            Example : [(0,1,0), (0,0,-1), (0,1,0), (0,0,1)] 
                    returns [(0,1,0), (0,1,0)]

            Example : [(0,1,0), (0,0,-1), (0,1,0), (0,0,1), (0,0,1), (0,0,1), (0,0,1)] 
                    returns [(0, 1, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1), (0, 0, 1)]

            Example : [(0,1,0)] * 8
                    returns [(0, -1, 0)] * 4

            Example : [(0,1,0)] * 8 + [(0,0,1)]
                    returns [(0, -1, 0)] * 4  + [(0,0,1)]

            Example : [(0,-1,0)] * 8 + [(0,0,1)]
                    returns [(0, 1, 0)] * 4  + [(0,0,1)]

            Example : [(0,1,0)] * 12
                    returns []
                    
            Example : [(0,1,0)] * 11 + [(0, 0, -1)] + [(0,1,0)] * 3
                    returns [(0, 1, 0), (0, 1, 0), (0, 0, -1)]
                    
            Example : [(0,1,0)] * 18 + [(0, 0, -1)]
                    returns [(0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 0, -1)
                    
            Example : [(0,1,0)] * 11 + [(0, 0, -1)] + [(0,1,0)] * 11 + [(0, 0, -1)] + [(0,1,0)] * 4
                    returns [(0, 1, 0), (0, 1, 0), (0, 0, -1), (0, 0, -1)]
            '''
            seq = np.array(l)
            head, ele = np.sum(seq, axis=0)[1:]
            
            # original sign 1 or -1 after summing
            h = 0 if head == 0 else int(head/abs(head))
            e = 0 if ele == 0 else int(ele/abs(ele))
            
            # min(7, 12-7)
            h_mul = abs(head) % 12
            e_mul = abs(ele)
            if h_mul > 6:
                h_mul = 12 - h_mul
                h = -h
            
            res_seq = [(0, h, 0)] * h_mul + [(0, 0, e)] * e_mul

            try:
                assert len(res_seq) <= 8
            except:
                print ("check compact rotation sequence length.")
                import pdb; pdb.set_trace()

            return res_seq

        return list(map(compress_single, rotation_history))

    def _rotates(self, rotation_instruction_batch):
        '''
        Sims take action using env level instructions. Modify rotation history as well.

        rotation_instruction_batch: list length=batch_size. 
                     each can be a single env action tuple such as (0,1,0) or a double action tuple such as ((0,0,-1), (0,0,-1))
        '''
        assert len(self.sims) == len(rotation_instruction_batch)

        for i, sim in enumerate(self.sims):
            rotations = rotation_instruction_batch[i]
            # there can be more than 1 sim-level rotation in a rotation instruction
            # ((0,0,-1), (0,0,-1))
            # or (0,0,-1)
            if isinstance(rotations[0], tuple):
                for index, heading, elevation in rotations:
                    # sim.makeAction(index, heading, elevation)
                    sim.makeAction(index, heading, elevation)
                    self._rotation_history[i].append((index, heading, elevation))
            else:
                index, heading, elevation = rotations
                # sim.makeAction(index, heading, elevation)
                sim.makeAction(index, heading, elevation)
                self._rotation_history[i].append((index, heading, elevation))        

    def _check_center(self, closest_vertex):
        '''within 30 deg horizontally and vertically)'''
        if (closest_vertex.rel_heading < math.pi/12.0 and closest_vertex.rel_heading > -math.pi/12.0) and \
        (closest_vertex.rel_elevation < math.pi/12.0 and closest_vertex.rel_elevation > -math.pi/12.0):
            return True
            
    def _check_above(self, closest_vertex):
        '''within 30 deg horizontally and more than 15 deg above'''
        if (closest_vertex.rel_heading < math.pi/12.0 and closest_vertex.rel_heading > -math.pi/12.0) and \
        (closest_vertex.rel_elevation > math.pi/12.0):
            return True
            
    def _check_below(self, closest_vertex):
        '''within 30 deg horizontally and more than 15 deg below'''
        if (closest_vertex.rel_heading < math.pi/12.0 and closest_vertex.rel_heading > -math.pi/12.0) and \
        (closest_vertex.rel_elevation < -math.pi/12.0):
            return True

    def _retrieve_curr_viewix_and_next_vertex(self):
        '''
        Returns:
            viewix_batch: list of integers, length = batch_size.
            closest_vertex_batch: list of viewpointId strings, length = batch_size.
        '''
        next_vertex_batch = [''] * self.batch_size 
        # list len=batch_size, each int
        # list len=batch_size, each a dictionary
        viewix_batch, closest_navigable_locations_batch = self._getStates()

        # check if whether we are looking at the closest navigable vertex directly.
        for i, closest_vertex in enumerate(closest_navigable_locations_batch):
            # only a few rotation angles are connected to another vertex
            if closest_vertex:
                # 1) directly facing -- within 30deg of center of view
                # 2,3) already looking up/down, but vertex is more than 15 deg further up/down
                if 	self._check_center(closest_vertex) or \
                    (viewix_batch[i] // 12 == 2 and self._check_above(closest_vertex)) or \
                    (viewix_batch[i] // 12 == 0 and self._check_below(closest_vertex)):
                    next_vertex_batch[i] = closest_vertex.viewpointId
                # otherwise, remain ''

        # list length batch_size, list length batch_size
        return viewix_batch, next_vertex_batch

    def _take_explore_step(self, rotation_instruction=None):
        '''
        1. Execute a single or no rotation instruction and collect the viewIndex
        2. Check if there's a reachable vertex within center of view, return the closest one if so. See definition of a reachable vertex in self._retrieve_curr_viewix_and_next_vertex.

        rotation_instruction : list len = batch_size, each element is either a single action tuple e.g.(0,1,0), or double action tuple e.g.((0,0,-1), (0,0,-1)).

        Returns: 
            A list of tuples. list len = batch_size.
            Each element in the returned list is a tuple with signature:
            (view_ix integer, list of env action tuples, vertex string)
        '''
        if rotation_instruction is None:
            # env skip rotation

            # list length batch_size (int), list length batch_size (str)
            viewix_batch, next_vertex_batch = self._retrieve_curr_viewix_and_next_vertex()
            efficient_rotations_batch = [[self.env_nav_actions[self.agent_nav_actions.index('<ignore>')]] for _ in range(self.batch_size)]
        else:
            # rotate and update rotation history
            self._rotates(rotation_instruction)  
            viewix_batch, next_vertex_batch = self._retrieve_curr_viewix_and_next_vertex()
            efficient_rotations_batch = self._map_to_efficient_rotation(self._rotation_history)
            
        assert len(efficient_rotations_batch) == len(viewix_batch) == len(next_vertex_batch) == self.batch_size

        # length = batch_size, [(view_ix, efficient actions, vertex), ...]
        return list(zip(viewix_batch, efficient_rotations_batch, next_vertex_batch))

#---------------------------------
    def _update_action_and_vertex_maps(self, viewix_env_actions_map, viewix_next_vertex_map, 
    viewix_actions_vertex_tuples):
        '''
        viewix_actions_vertex_tuples:   zipped object. length = batch_size. 
                                        e.g. [(view_ix, efficient actions, vertex), ...]
        viewix_env_actions_map: list (batch_size, 36, varies). Each [(0,1,0), (0,-1,1), ...]
        viewix_next_vertex_map: list (batch_size, 36). Each string.
        '''
        # loop through batch
        for i, (viewix, env_actions_list, vertex_str) in enumerate(viewix_actions_vertex_tuples):
            
            assert isinstance(env_actions_list, list)
            viewix_env_actions_map[i][viewix] = env_actions_list
            viewix_next_vertex_map[i][viewix] = vertex_str
        return viewix_env_actions_map, viewix_next_vertex_map  

    def _explore_horizontally(self, viewix_env_actions_map, viewix_next_vertex_map, heading_adjusts, timestep=None):
        '''
        Turn around horizontally and look- 11 times

        viewix_env_actions_map: list (batch_size, 36, varies). Each [(0,1,0), (0,-1,1), ...]
        viewix_next_vertex_map: list (batch_size, 36, varies). Each string.
        '''
        for _ in range(11):
            # list, length = batch_size, [(view_ix, efficient actions, vertex), ...]
            viewix_actions_vertex_tuples = self._take_explore_step(heading_adjusts)
            viewix_env_actions_map, viewix_next_vertex_map = self._update_action_and_vertex_maps( 
                viewix_env_actions_map, 
                viewix_next_vertex_map, 
                viewix_actions_vertex_tuples)
        return viewix_env_actions_map, viewix_next_vertex_map

    def _explore_elevation(self, viewix_env_actions_map, viewix_next_vertex_map, elevation_adjusts, timestep=None):
        '''
        Look up or down to get observe.

        viewix_env_actions_map: list (batch_size, 36, varies). Each [(0,1,0), (0,-1,1), ...]
        viewix_next_vertex_map: list (batch_size, 36, varies). Each string.
        '''
        # list, length = batch_size, [(view_ix, efficient actions, vertex), ...]
        viewix_actions_vertex_tuples = self._take_explore_step(elevation_adjusts) 
        return self._update_action_and_vertex_maps(viewix_env_actions_map, viewix_next_vertex_map, \
            viewix_actions_vertex_tuples)

    def _explore_current(self, viewix_env_actions_map, viewix_next_vertex_map, timestep=None):
        '''
        Observe from current observation angle, without further rotation.

        viewix_env_actions_map: list (batch_size, 36, varies). Each [(0,1,0), (0,-1,1), ...]
        viewix_next_vertex_map: list (batch_size, 36, varies). Each string.
        '''
        # list, length = batch_size, 
        # i.e. [(view_ix int, efficient actions, vertex str), ...]
        viewix_actions_vertex_tuples = self._take_explore_step(None)
        return self._update_action_and_vertex_maps(viewix_env_actions_map, viewix_next_vertex_map, \
            viewix_actions_vertex_tuples)

    def explore_sphere(self, explore_instructions, sphere_size, timestep=None):
        '''
        Batch of simulators look around their panoramic spheres and collect 3 mappings:
        - viewIndex to env level rotation action
        - viewIndex to next vertex viewpoint Id
        - viewIndex to 1/0 indicating if the rotation angle directly faces a reachable next vertex.

        explore_instructions : a tuple of 3 lists. each length = batch_size. each element in respective list is either a single action tuple e.g.(0,1,0), or double action tuple e.g.((0,0,-1), (0,0,-1)). 

        sphere_size : int = 36 -- from hparams. There are 36 discretized images in an agent's panoramic sphere.

        Returns:
            viewix_env_actions_map: list (batch_size, 36, varies). Each [(0,1,0), (0,-1,1), ...]

            viewix_next_vertex_map: list (batch_size, 36). Each string.

            view_index_mask: array shape (36, batch_size). True if a particular view angle doesn't have any directly reachable next vertex on the floor plan graph.
        ''' 
        viewix_env_actions_map = [[None for s in range(sphere_size)] for _ in range(self.batch_size)]
        viewix_next_vertex_map = [[None for s in range(sphere_size)] for _ in range(self.batch_size)]

        # elaborate condensed exploration instructions into 35 steps that cover the sphere
        heading_adjusts, elevation_adjusts_1, elevation_adjusts_2 = explore_instructions

        # capture current - 1 time
        viewix_env_actions_map, viewix_next_vertex_map = \
            self._explore_current(viewix_env_actions_map, viewix_next_vertex_map, timestep)

        # turn around horizontally - 11 times
        viewix_env_actions_map, viewix_next_vertex_map = \
            self._explore_horizontally(viewix_env_actions_map, viewix_next_vertex_map, heading_adjusts, timestep)

        # adjust elevation - 1 time
        viewix_env_actions_map, viewix_next_vertex_map = \
            self._explore_elevation(viewix_env_actions_map, viewix_next_vertex_map, elevation_adjusts_1)

        # turn around horizontally - 11 times
        viewix_env_actions_map, viewix_next_vertex_map = \
            self._explore_horizontally(viewix_env_actions_map, viewix_next_vertex_map, heading_adjusts)

        # adjust elevation - 1 time
        viewix_env_actions_map, viewix_next_vertex_map = \
            self._explore_elevation(viewix_env_actions_map, viewix_next_vertex_map, elevation_adjusts_2)

        # turn around horizontally - 11 times
        viewix_env_actions_map, viewix_next_vertex_map = \
            self._explore_horizontally(viewix_env_actions_map, viewix_next_vertex_map, heading_adjusts)

        # Replace None in env actions and next vertex if an ob has ended
        # loop through batch
        for i in range(len(heading_adjusts)):  # 0 - 100
            if heading_adjusts[i] == elevation_adjusts_1[i] == elevation_adjusts_2[i] == self.env_nav_actions[self.agent_nav_actions.index('<ignore>')]:
                for j in range(sphere_size):  # 0 - 35
                    if viewix_env_actions_map[i][j] is None:
                        assert viewix_next_vertex_map[i][j] is None
                        viewix_env_actions_map[i][j] = [(0,0,0)]
                        viewix_next_vertex_map[i][j] = ''

        # assert all 36 views are covered for each ob in the batch (no Nones at all)
        assert all([v_str is not None for task_sphere in viewix_next_vertex_map for v_str in task_sphere])

        # arr shape(36, batch_size), each boolean
        view_index_mask = (np.array(viewix_next_vertex_map) == '').transpose()
        assert view_index_mask.shape == (sphere_size, self.batch_size)

        return viewix_env_actions_map, viewix_next_vertex_map, view_index_mask


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
        self.traj_len_ref = 'paths' if hparams.navigation_objective == 'value_estimation' \
            else 'trajectories'

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
                len(t) for t in item[self.traj_len_ref])

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
                    for t in item[self.traj_len_ref]) / len(item[self.traj_len_ref])
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
        multiple_actions : list/tuple of len=batch_size
                           each element is a list of env action tuples
        '''
        assert len(self.env.sims) == len(multiple_actions)
        for i in range(len(self.env.sims)):
            # a list/tuple of tuples
            actions = multiple_actions[i]
            for index, heading, elevation in actions:
                self.env.sims[i].makeAction(index, heading, elevation)
        return self._get_obs()

    def prepend_instruction(self, idx, instr):
        ''' Prepend subgoal to end-goal. '''

        self.instructions[idx] = instr + ' . ' + self.batch[idx]['instruction']

    def get_obs(self):
        return self._get_obs()



