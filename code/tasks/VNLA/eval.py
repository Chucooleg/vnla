import json
import os
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score
import pprint
pp = pprint.PrettyPrinter(indent=4)

from utils import load_datasets, load_nav_graphs, load_region_label_to_name, load_panos_to_region


class Evaluation(object):

    def __init__(self, hparams, splits, data_path):
        self.success_radius = hparams.success_radius
        self.splits = splits

        self.scans = set()
        self.graphs = {}
        self.distances = {}

        self.no_room = hasattr(hparams, 'no_room') and hparams.no_room
        if splits:
            self.load_data(load_datasets(
                splits=splits, 
                path=data_path,
                prefix='noroom' if self.no_room else 'asknav',
                suffix=hparams.data_suffix if hasattr(hparams, 'data_suffix') else ''))

        self.region_label_to_name = load_region_label_to_name()
        self.panos_to_region = {}
        for scan in self.scans:
            self.panos_to_region[scan] = load_panos_to_region(scan, self.region_label_to_name)


    def load_data(self, data):
        self.gt = {}
        self.instr_ids = []
        scans = []
        for item in data:
            self.gt[str(item['path_id'])] = item
            if isinstance(item['path_id'], int):
                self.instr_ids.extend(['%d_%d' % (item['path_id'],i)
                    for i in range(len(item['instructions']))])
            else:
                self.instr_ids.extend(['%s_%d' % (item['path_id'],i)
                    for i in range(len(item['instructions']))])
            scans.append(item['scan'])
        self.instr_ids = set(self.instr_ids)
        scans = set(scans)

        new_scans = set.difference(scans, self.scans)
        if new_scans:
            for scan in new_scans:
                self.graphs[scan] = load_nav_graphs(scan)
                self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(self.graphs[scan]))
        self.scans.update(new_scans)

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path):
        gt = self.gt[instr_id[:instr_id.rfind('_')]]
        scan = gt['scan']

        self.scores['instr_id'].append(instr_id)
        self.scores['trajectory_steps'].append(len(path) - 1)

        nav_errors = oracle_errors = 1e9
        for shortest_path in gt['paths']:
            start = shortest_path[0]
            assert start == path[0][0], 'Result trajectories should include the start position'
            goal = shortest_path[-1]
            final_pos = path[-1][0]
            nearest_pos = self._get_nearest(scan, goal, path)
            nav_errors = min(nav_errors, self.distances[scan][final_pos][goal])
            oracle_errors = min(oracle_errors, self.distances[scan][nearest_pos][goal])

        self.scores['nav_errors'].append(nav_errors)
        self.scores['oracle_errors'].append(oracle_errors)
        distance = 0
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[scan][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)

        if not self.no_room:
            goal_room = None
            for shortest_path in gt['paths']:

                assert goal_room is None or goal_room == \
                    self.panos_to_region[scan][shortest_path[-1]]
                goal_room = self.panos_to_region[scan][shortest_path[-1]]

            assert goal_room is not None
            final_room = self.panos_to_region[scan][path[-1][0]]
            self.scores['room_successes'].append(final_room == goal_room)

    def check_success(self, d):
        return d <= self.success_radius

    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        with open(output_file) as f:
            for item in json.load(f):
                # Check against expected ids
                if item['instr_id'] in instr_ids:
                    instr_ids.remove(item['instr_id'])
                    self._score_item(item['instr_id'], item['trajectory'])

        # TODO: uncomment/comment this block on with large/small data!
        assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s'\
                       % (len(instr_ids), len(self.instr_ids), ",".join(self.splits), output_file)
        assert len(self.scores['nav_errors']) == len(self.instr_ids)

        score_summary = {
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle_error': np.average(self.scores['oracle_errors']),
            'steps': np.average(self.scores['trajectory_steps']),
            'length': np.average(self.scores['trajectory_lengths'])
        }
        is_success = [(instr_id, self.check_success(d)) for d, instr_id
            in zip(self.scores['nav_errors'], self.scores['instr_id'])]
        num_successes = len([d for d in self.scores['nav_errors'] if self.check_success(d)])
        score_summary['success_rate'] = float(num_successes)/float(len(self.scores['nav_errors']))
        oracle_successes = len([d for d in self.scores['oracle_errors'] if self.check_success(d)])
        score_summary['oracle_rate'] = float(oracle_successes)/float(len(self.scores['oracle_errors']))
        if not self.no_room:
            score_summary['room_success_rate'] = float(sum(self.scores['room_successes'])) / \
                len(self.scores['room_successes'])
        return score_summary, self.scores, is_success


class SwapEvaluation(object):

    def __init__(self, hparams, split, data_path):
        self.split = split

        self.data_path = os.path.join(hparams.data_path, 'asknav_pretrain_lookup_{}.json'.format(split))

        self.load_instr_ids()

    def load_instr_ids(self):
        '''load the validation dataset'''
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        self.instr_ids = [item['instr_id'] for item in data]

    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x)) 

    def score(self, output_file):
        '''Return an overall accuracy and number of datapoints'''
        instr_ids = set(self.instr_ids)
        self.pred_probs = []
        self.targets = []

        with open(output_file, 'r') as f:
            output = json.load(f)
            for item in output:
                if item['instr_id'] in instr_ids:
                    instr_ids.remove(item['instr_id'])
                    self.pred_probs.append(self._sigmoid(item['logit']))
                    self.targets.append(item['target'])
        
        assert len(instr_ids) == 0 'Missing %d of %d instruction ids from %s - not in %s'\
                       % (len(instr_ids), len(self.instr_ids), ",".join(self.split), output_file)

        self.pred_probs = np.array(self.pred_probs)
        self.targets = np.array(self.targets)
        # scalar
        auc = roc_auc_score(y_true=self.targets, y_score=self.pred_probs)
        thresholded_labs = self.pred_probs > 0.5
        accuracy = np.mean(thresholded_labs == self.targets)

        return auc, accuracy, len(self.pred_probs), self.pred_probs, self.targets


def compute_auc(res):
    '''
    res : dictionary. res['preds'][env_name] gives an (N, ) array of prediction prob. res['tars'][env_name] gives an (N, ) array of prediction targets. 

    Returns: scalar auc value
    '''
    y_score = np.hstack(res['preds']['val_seen'], res['preds']['val_unseen'])
    y_true = np.hstack(res['tars']['val_seen'], res['tars']['val_unseen'])
    assert y_score.shape == y_true.shape
    return roc_auc_score(y_true=y_true, y_score=y_score)





