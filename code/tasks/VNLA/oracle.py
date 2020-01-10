# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import math
import networkx as nx
import functools
import scipy.stats
import random
import sys
import copy
import numpy as np

import torch

import utils
sys.path.append('../../build')
import MatterSim

class ShortestPathOracle(object):
    ''' Shortest navigation teacher '''

    def __init__(self, agent_nav_actions, env_nav_actions=None):
        self.scans = set()
        self.graph = {}
        self.paths = {}
        self.distances = {}
        self.agent_nav_actions = agent_nav_actions

        if env_nav_actions is not None:
            self.env_nav_actions = env_nav_actions

    def add_scans(self, scans, path=None):
        new_scans = set.difference(scans, self.scans)
        if new_scans:
            print('Loading navigation graphs for %d scans' % len(new_scans))
            for scan in new_scans:
                graph, paths, distances = self._compute_shortest_paths(scan, path=path)
                self.graph[scan] = graph
                self.paths[scan] = paths
                self.distances[scan] = distances
            self.scans.update(new_scans)

    def _compute_shortest_paths(self, scan, path=None):
        ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
        graph = utils.load_nav_graphs(scan, path=path)
        paths = dict(nx.all_pairs_dijkstra_path(graph))
        distances = dict(nx.all_pairs_dijkstra_path_length(graph))
        return graph, paths, distances

    def _find_nearest_point(self, scan, start_point, end_points):
        best_d = 1e9
        best_point = None

        for end_point in end_points:
            d = self.distances[scan][start_point][end_point]
            if d < best_d:
                best_d = d
                best_point = end_point
        return best_d, best_point

    def _find_nearest_point_on_a_path(self, scan, current_point, start_point, goal_point):
        path = self.paths[scan][start_point][goal_point]
        return self._find_nearest_point(scan, current_point, path)

    def _shortest_path_action(self, ob):
        ''' Determine next action on the shortest path to goals. '''

        scan = ob['scan']
        start_point = ob['viewpoint']

        # Find nearest goal
        _, goal_point = self._find_nearest_point(scan, start_point, ob['goal_viewpoints'])

        # Stop if a goal is reached
        if start_point == goal_point:
            return (0, 0, 0)

        path = self.paths[scan][start_point][goal_point]
        next_point = path[1]

        # Can we see the next viewpoint?
        for i, loc in enumerate(ob['navigableLocations']):
            if loc.viewpointId == next_point:
                # Look directly at the viewpoint before moving
                if loc.rel_heading > math.pi/6.0:
                      return (0, 1, 0) # Turn right
                elif loc.rel_heading < -math.pi/6.0:
                      return (0,-1, 0) # Turn left
                elif loc.rel_elevation > math.pi/6.0 and ob['viewIndex'] // 12 < 2:
                      return (0, 0, 1) # Look up
                elif loc.rel_elevation < -math.pi/6.0 and ob['viewIndex'] // 12 > 0:
                      return (0, 0,-1) # Look down
                else:
                      return (i, 0, 0) # Move

        # Can't see it - first neutralize camera elevation
        if ob['viewIndex'] // 12 == 0:
            return (0, 0, 1) # Look up
        elif ob['viewIndex'] // 12 == 2:
            return (0, 0,-1) # Look down

        # If camera is already neutralized, decide which way to turn
        target_rel = self.graph[ob['scan']].node[next_point]['position'] - ob['point']  # state.location.point
        # 180deg - 
        target_heading = math.pi / 2.0 - math.atan2(target_rel[1], target_rel[0])
        if target_heading < 0:
            target_heading += 2.0 * math.pi
        if ob['heading'] > target_heading and ob['heading'] - target_heading < math.pi:
            return (0, -1, 0) # Turn left
        if target_heading > ob['heading'] and target_heading - ob['heading'] > math.pi:
            return (0, -1, 0) # Turn left

        return (0, 1, 0) # Turn right

    def _map_env_action_to_agent_action(self, action, ob):
        ix, heading_chg, elevation_chg = action
        if heading_chg > 0:
            return self.agent_nav_actions.index('right')
        if heading_chg < 0:
            return self.agent_nav_actions.index('left')
        if elevation_chg > 0:
            return self.agent_nav_actions.index('up')
        if elevation_chg < 0:
            return self.agent_nav_actions.index('down')
        if ix > 0:
            return self.agent_nav_actions.index('forward')
        if ob['ended']:
            return self.agent_nav_actions.index('<ignore>')
        return self.agent_nav_actions.index('<end>')

    def interpret_agent_action(self, action_idx, ob):
        '''Translate action index back to env action for simulator to take'''

        # If the action is not `forward`, simply map it to the simulator's
        # action space
        if action_idx != self.agent_nav_actions.index('forward'):
            return self.env_nav_actions[action_idx]

        # If the action is forward, more complicated
        scan = ob['scan']
        start_point = ob['viewpoint']

        # Find nearest goal view point
        _, goal_point = self._find_nearest_point(scan, start_point, ob['goal_viewpoints'])

        optimal_path = self.paths[scan][start_point][goal_point]

        # If the goal is right in front of us, go to it.
        # The dataset guarantees that the goal is always reachable.
        if len(optimal_path) < 2:
            return (1, 0, 0)

        next_optimal_point = optimal_path[1]

        # If the next optimal viewpoint is within 30 degrees of
        # the center of the view, go to it.
        for i, loc in enumerate(ob['navigableLocations']):
            if loc.viewpointId == next_optimal_point:
                if loc.rel_heading > math.pi/6.0 or loc.rel_heading < -math.pi/6.0 or \
                   (loc.rel_elevation > math.pi/6.0  and ob['viewIndex'] // 12 < 2) or \
                   (loc.rel_elevation < -math.pi/6.0 and ob['viewIndex'] // 12 > 0):
                    continue
                else:
                    return (i, 0, 0)

        # Otherwise, go the navigable (seeable) viewpt that has the least angular distance from the center of the current image (viewpt).
        return (1, 0, 0)

    def __call__(self, obs):
        self.actions = list(map(self._shortest_path_action, obs))
        return list(map(self._map_env_action_to_agent_action, self.actions, obs))


class FrontierShortestPathsOracle(ShortestPathOracle):

    def __init__(self, agent_nav_actions, env_nav_actions=None):
        super(FrontierShortestPathsOracle, self).__init__(agent_nav_actions, env_nav_actions)

        # self.env_nav_actions = env_nav_actions
        self.valid_rotation_action_indices = [self.agent_nav_actions.index(r) for r in ('left', 'right', 'up', 'down', '<ignore>')]

    # inherit parent add_scans() function

    def interpret_agent_rotations(self, rotation_action_indices, ob):
        '''
        rotation_action_indices : a list of int action indices
        Returns:
            list of fixed length agent.max_macro_action_seq_len (e.g. 8)
            e.g. [(0, 1, 0), (0, 1, -1), ..... (0,0,0)]
            e.g. [(0,0,0), ... (0,0,0)] if ob has ended.
        '''
        max_macro_action_seq_len = len(rotation_action_indices)
        macro_rotations = [self.agent_nav_actions.index('<ignore>')] * max_macro_action_seq_len
        if not ob['ended']:
            for i, action_idx in enumerate(rotation_action_indices):
                assert action_idx in self.valid_rotation_action_indices
                macro_rotations[i] = self.env_nav_actions[action_idx]
        return macro_rotations

    def interpret_agent_forward(self, ob):
        '''
        Returns:
            (0, 0, 0) to ignore if trajectory has already ended
            or
            (1, 0, 0) to step forward to the direct facing vertex
        '''
        if ob['ended']:
            return self.env_nav_actions[self.agent_nav_actions.index('<ignore>')]
        else:
            return self.env_nav_actions[self.agent_nav_actions.index('forward')]

    def make_explore_instructions(self, obs):
        '''
        Make env level rotation instructions of each ob to explore its own panoramic sphere. The output should be informative enough for agent to collect information from all 36 facets of its panoramic sphere.

        Returns:
        heading_adjusts:     list len=batch_size, each an env action tuple.
        elevation_adjusts_1: same.
        elevation_adjusts_2: list len=batch_size, each either a single action tuple e.g.(0,1,0), or double action tuple e.g.((0,0,-1), (0,0,-1)).
        '''
        batch_size = len(obs)

        # How agent explore the entire pano sphere
        # Right*11, Up/Down, Right*11, Up/Down (*2), Right*11
        heading_adjusts = [()] * batch_size
        elevation_adjusts_1 = [()] * batch_size
        elevation_adjusts_2 = [()] * batch_size

        # (0,0,1)
        up_tup = self.env_nav_actions[self.agent_nav_actions.index('up')]
        # (0,0,-1)
        down_tup = self.env_nav_actions[self.agent_nav_actions.index('down')]
        # (0,1,0)
        right_tup = self.env_nav_actions[self.agent_nav_actions.index('right')]
        # (0,0,0)
        ignore_tup = self.env_nav_actions[self.agent_nav_actions.index('<ignore>')]

        # Loop through each ob in the batch
        for i, ob in enumerate(obs):
            if ob['ended']:
                # don't move at all.
                heading_adjusts[i] = ignore_tup
                elevation_adjusts_1[i] = ignore_tup
                elevation_adjusts_2[i] = ignore_tup
            else:
                # turn right for 11 times at every elevation level.
                heading_adjusts[i] = right_tup
                # check agent elevation
                if ob['viewIndex'] // 12 == 0:
                    # facing down, so need to look up twice.
                    elevation_adjusts_1[i] = elevation_adjusts_2[i] = up_tup
                elif ob['viewIndex'] // 12 == 2:
                    # facing up, so need to look down twice.
                    elevation_adjusts_1[i] = elevation_adjusts_2[i] = down_tup
                else:  
                    # neutral, so need to look up once, and then look down twice
                    elevation_adjusts_1[i] = up_tup
                    elevation_adjusts_2[i] = (down_tup, down_tup)

        return heading_adjusts, elevation_adjusts_1, elevation_adjusts_2

    def compute_frontier_cost_single(self, ob, next_viewpoint_index_str):
        '''
        next_viewpoint_index_str: single str indicating viewpoint index. 
                                  e.g. '1e6b606b44df4a6086c0f97e826d4d15'
        '''
        cost_stepping = self.distances[ob['scan']][ob['viewpoint']][next_viewpoint_index_str]
        cost_togo, _ = self._find_nearest_point(ob['scan'], ob['viewpoint'], ob['goal_viewpoints'])
        assert cost_stepping > 0 and cost_togo > 0
        return cost_togo + cost_stepping

    def compute_frontier_costs(self, obs, viewix_next_vertex_map):
        '''
        For each ob, compute:
        cost = cost-to-go + cost-stepping for all reachable vertices
        '''
        assert len(obs) == len(viewix_next_vertex_map)
        # arr shape (batch_size, 36)
        q_values_target = np.ones(len(obs), len(viewix_next_vertex_map[0])) * 1e9
        # Loop through batch
        for i, ob in enumerate(obs):
            # NOTE ended ob won't be added to hist buffer for training
            if not ob['ended']:
                costs = []
                for proposed_vertex in viewix_next_vertex_map[i]:
                    if proposed_vertex == '':
                        costs.append(1e9)
                    else:
                        # add up cost-togo + cost-stepping
                        costs.append(self.compute_frontier_cost_single(ob, proposed_vertex))
                assert len(costs) == len(viewix_next_vertex_map[0])  # 36
                q_values_target[i, :] = costs
        return q_values_target

    def _map_env_action_to_agent_action(self, action, ob):
        '''
        Translate rotation env action seq into agent action index seq.
        '''
        if ob['ended']:
            return self.agent_nav_actions.index('<ignore>')

        ix, heading_chg, elevation_chg = action

        assert ix == 0, 'Accept only rotation or ignore actions'
        assert heading_chg == 0 or elevation_chg == 0, 'Accept only one rotation action at a time'
        
        if heading_chg > 0:
            return self.agent_nav_actions.index('right')
        if heading_chg < 0:
            return self.agent_nav_actions.index('left')
        if elevation_chg > 0:
            return self.agent_nav_actions.index('up')
        if elevation_chg < 0:
            return self.agent_nav_actions.index('down')        

    def translate_env_actions(self, obs, viewix_env_actions_map, max_macro_action_seq_len, sphere_size):
        '''
        viewix_env_actions_map : list (batch_size, 36, varies). Each [(0,1,0), (0,0,-1), ...]

        Returns:
            viewix_actions_map : array shape(36, batch_size, self.max_macro_action_seq_len)
        '''
        # tensor shape(36, batch_size, self.max_macro_action_seq_len)
        viewix_actions_map = np.ones((sphere_size, len(obs), max_macro_action_seq_len), dtype='int') * \
            self.agent_nav_actions.index('<ignore>')

        for i, ob in enumerate(obs): # 1-100
            for j, env_action_tup_seq in viewix_env_actions_map[i]: # 1-36
                assert len(env_action_tup_seq) <= 8
                # map seq, length varies
                agent_action_seq = list(map(self._map_env_action_to_agent_action, env_action_tup_seq, ob))
                assert len(agent_action_seq) <= 8
                # assign action index, seq is already padded to 8 during initialization
                viewix_actions_map[j, i, :len(agent_action_seq)] = agent_action_seq

        return viewix_actions_map


class AskOracle(object):

    DONT_ASK = 0
    ASK = 1

    def __init__(self, hparams, agent_ask_actions):
        self.deviate_threshold = hparams.deviate_threshold
        self.uncertain_threshold = hparams.uncertain_threshold
        self.unmoved_threshold = hparams.unmoved_threshold
        self.agent_ask_actions = agent_ask_actions

        self.rule_a_e = hasattr(hparams, 'rule_a_e') and hparams.rule_a_e
        self.rule_b_d = hasattr(hparams, 'rule_b_d') and hparams.rule_b_d

    def _should_ask_rule_a_e(self, ob, nav_oracle=None):

        if ob['queries_unused'] <= 0:
            return self.DONT_ASK, 'exceed'

        scan = ob['scan']
        current_point = ob['viewpoint']
        _, goal_point = nav_oracle._find_nearest_point(scan, current_point, ob['goal_viewpoints'])

        agent_decision = int(np.argmax(ob['nav_dist']))
        if current_point == goal_point and \
           agent_decision == nav_oracle.agent_nav_actions.index('forward'):
            return self.ASK, 'arrive'

        start_point = ob['init_viewpoint']
        d, _ = nav_oracle._find_nearest_point_on_a_path(scan, current_point, start_point, goal_point)
        if d > self.deviate_threshold:
            return self.ASK, 'deviate'

        return self.DONT_ASK, 'pass'

    def _should_ask_rule_b_d(self, ob, nav_oracle=None):

        if ob['queries_unused'] <= 0:
            return self.DONT_ASK, 'exceed'

        agent_dist = ob['nav_dist']
        uniform = [1. / len(agent_dist)] * len(agent_dist)
        entropy_gap = scipy.stats.entropy(uniform) - scipy.stats.entropy(agent_dist)
        if entropy_gap < self.uncertain_threshold - 1e-9:
            return self.ASK, 'uncertain'

        if len(ob['agent_path']) >= self.unmoved_threshold:
            last_nodes = [t[0] for t in ob['agent_path']][-self.unmoved_threshold:]
            if all(node == last_nodes[0] for node in last_nodes):
                return self.ASK, 'unmoved'

        if ob['queries_unused'] >= ob['traj_len'] - ob['time_step']:
            return self.ASK, 'why_not'

        return self.DONT_ASK, 'pass'

    def _should_ask(self, ob, nav_oracle=None):

        if self.rule_a_e:
            return self._should_ask_rule_a_e(ob, nav_oracle=nav_oracle)

        if self.rule_b_d:
            return self._should_ask_rule_b_d(ob, nav_oracle=nav_oracle)

        if ob['queries_unused'] <= 0:
            return self.DONT_ASK, 'exceed'

        # Find nearest point on the current shortest path
        scan = ob['scan']
        current_point = ob['viewpoint']
        # Find nearest goal to current point
        _, goal_point = nav_oracle._find_nearest_point(scan, current_point, ob['goal_viewpoints'])

        # Rule (e): ask if the goal has been reached but the agent decides to
        # go forward
        agent_decision = int(np.argmax(ob['nav_dist']))
        if current_point == goal_point and \
           agent_decision == nav_oracle.agent_nav_actions.index('forward'):
            return self.ASK, 'arrive'

        start_point = ob['init_viewpoint']
        # Find closest point to the current point on the path from start point
        # to goal point
        d, _ = nav_oracle._find_nearest_point_on_a_path(scan, current_point,
            start_point, goal_point)
        # Rule (a): ask if the agent deviates too far from the optimal path
        if d > self.deviate_threshold:
            return self.ASK, 'deviate'

        # Rule (b): ask if uncertain
        agent_dist = ob['nav_dist']
        uniform = [1. / len(agent_dist)] * len(agent_dist)
        entropy_gap = scipy.stats.entropy(uniform) - scipy.stats.entropy(agent_dist)
        if entropy_gap < self.uncertain_threshold - 1e-9:
            return self.ASK, 'uncertain'

        # Rule (c): ask if not moving for too long
        if len(ob['agent_path']) >= self.unmoved_threshold:
            last_nodes = [t[0] for t in ob['agent_path']][-self.unmoved_threshold:]
            if all(node == last_nodes[0] for node in last_nodes):
                return self.ASK, 'unmoved'

        # Rule (d): ask to spend all budget at the end
        if ob['queries_unused'] >= ob['traj_len'] - ob['time_step']:
            return self.ASK, 'why_not'

        return self.DONT_ASK, 'pass'

    def _map_env_action_to_agent_action(self, action, ob):
        if ob['ended']:
            return self.agent_ask_actions.index('<ignore>')
        if action == self.DONT_ASK:
            return self.agent_ask_actions.index('dont_ask')
        return self.agent_ask_actions.index('ask')

    def __call__(self, obs, nav_oracle):
        should_ask_fn = functools.partial(self._should_ask, nav_oracle=nav_oracle)
        actions, reasons = zip(*list(map(should_ask_fn, obs)))
        actions = list(map(self._map_env_action_to_agent_action, actions, obs))
        return actions, reasons


class MultistepShortestPathOracle(ShortestPathOracle):
    '''For Ask Agents with direct advisors'''

    def __init__(self, n_steps, agent_nav_actions, env_nav_actions):
        super(MultistepShortestPathOracle, self).__init__(agent_nav_actions)
        self.sim = MatterSim.Simulator()
        self.sim.setRenderingEnabled(False)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setCameraResolution(640, 480)
        self.sim.setCameraVFOV(math.radians(60))
        self.sim.setNavGraphPath(
            os.path.join(os.getenv('PT_DATA_DIR'), 'connectivity'))
        self.sim.init()
        self.n_steps = n_steps
        self.env_nav_actions = env_nav_actions

    def _shortest_path_actions(self, ob):
        actions = []
        self.sim.newEpisode(ob['scan'], ob['viewpoint'], ob['heading'], ob['elevation'])

        assert not ob['ended']

        for _ in range(self.n_steps):
            # Query oracle for next action
            action = self._shortest_path_action(ob)
            # Convert to agent action
            agent_action = self._map_env_action_to_agent_action(action, ob)
            actions.append(agent_action)
            # Take action
            self.sim.makeAction(*action)

            if action == (0, 0, 0):
                break

            state = self.sim.getState()
            ob = {
                    'viewpoint': state.location.viewpointId,
                    'viewIndex': state.viewIndex,
                    'heading'  : state.heading,
                    'elevation': state.elevation,
                    'navigableLocations': state.navigableLocations,
                    'point'    : state.location.point,
                    'ended'    : ob['ended'] or action == (0, 0, 0),
                    'goal_viewpoints': ob['goal_viewpoints'],
                    'scan'     : ob['scan']
                }

        return actions

    def __call__(self, ob):
        return self._shortest_path_actions(ob)


class NextOptimalOracle(object):

    def __init__(self, hparams, agent_nav_actions, env_nav_actions,
                 agent_ask_actions):
        self.type = 'next_optimal'
        self.ask_oracle = make_oracle('ask', hparams, agent_ask_actions)
        self.nav_oracle = make_oracle('shortest', agent_nav_actions, env_nav_actions)

    def __call__(self, obs):
        ask_actions, ask_reasons = self.ask_oracle(obs, self.nav_oracle)

        self.nav_oracle.add_scans(set(ob['scan'] for ob in obs))
        nav_actions = self.nav_oracle(obs)

        return nav_actions, ask_actions, ask_reasons

    def add_scans(self, scans):
        self.nav_oracle.add_scans(scans)

    def next_ask(self, obs):
        return self.ask_oracle(obs, self.nav_oracle)

    def next_nav(self, obs):
        return self.nav_oracle(obs)

    def interpret_agent_action(self, *args, **kwargs):
        return self.nav_oracle.interpret_agent_action(*args, **kwargs)


class StepByStepSubgoalOracle(object):

    def __init__(self, n_steps, agent_nav_actions, env_nav_actions, mode=None):
        self.type = 'step_by_step'
        self.nav_oracle = make_oracle('direct', n_steps, agent_nav_actions, env_nav_actions)
        self.agent_nav_actions = agent_nav_actions
        if mode == 'easy':
            self._map_actions_to_instruction = self._map_actions_to_instruction_easy
        elif mode == 'hard':
            self._map_actions_to_instruction = self._map_actions_to_instruction_hard
        else:
            sys.exit('unknown step by step mode!')

    def add_scans(self, scans):
        self.nav_oracle.add_scans(scans)

    def _make_action_name(self, a):
        action_name = self.agent_nav_actions[a]
        if action_name in ['up', 'down']:
            return 'look ' + action_name
        elif action_name in ['left', 'right']:
            return 'turn ' + action_name
        elif action_name == 'forward':
            return 'go ' + action_name
        elif action_name == '<end>':
            return 'stop'
        elif action_name == '<ignore>':
            return ''
        return None

    def _map_actions_to_instruction_hard(self, actions):
        agg_actions = []
        cnt = 1
        for i in range(1, len(actions)):
            if actions[i] != actions[i - 1]:
                agg_actions.append((actions[i - 1], cnt))
                cnt = 1
            else:
                cnt += 1
        agg_actions.append((actions[-1], cnt))
        instruction = []
        for a, c in agg_actions:
            action_name = self._make_action_name(a)
            if c > 1:
                if 'turn' in action_name:
                    degree = 30 * c
                    if 'left' in action_name:
                        instruction.append('turn %d degrees left' % degree)
                    elif 'right' in action_name:
                        instruction.append('turn %d degrees right' % degree)
                    else:
                        raise(ValueError, action_name)
                elif 'go' in action_name:
                    instruction.append('%s %d steps' % (action_name, c))
            elif action_name != '':
                instruction.append(action_name)
        return ' , '.join(instruction)

    def _map_actions_to_instruction_easy(self, actions):
        instruction = []
        for a in actions:
            instruction.append(self._make_action_name(a))
        return ' , '.join(instruction)

    def __call__(self, ob):
        action_seq = self.nav_oracle(ob)
        verbal_instruction = self._map_actions_to_instruction(action_seq)
        return action_seq, verbal_instruction


def make_oracle(oracle_type, *args, **kwargs):
    if oracle_type == 'shortest':
        return ShortestPathOracle(*args, **kwargs)
    if oracle_type == 'next_optimal':
        return NextOptimalOracle(*args, **kwargs)
    if oracle_type == 'ask':
        return AskOracle(*args, **kwargs)

    if oracle_type == 'direct':
        return MultistepShortestPathOracle(*args, **kwargs)
    if oracle_type == 'verbal':
        return StepByStepSubgoalOracle(*args, **kwargs)

    if oracle_type == 'frontier_shortest':
        return FrontierShortestPathsOracle(*args, **kwargs)
    # TODO implement next
    # if oracle_type == 'diverse_shortest':
    #     return DiverseShortestPathsOracle(*args, **kwargs)

    return None
