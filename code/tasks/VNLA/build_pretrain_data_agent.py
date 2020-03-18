# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import sys
import os
import numpy as np
import argparse
from argparse import Namespace

from env import VNLABuildPretrainBatch

class BuildPretrainDataAgent():

    def __init__(self, hparams):
        self.results_path = hparams.results_path
        self.results = {}

    def write_results(self):
        output = []
        for k, v  in self.results.items():
            item = {'instr_id': k}
            item.update(v)
            output.append(item)

        with open(self.results_path, 'w') as f:
            try:
                json.dump(output, f)
            except:
                import pdb; pdb.set_trace()

    def _collect_pretrain_data(self, pretrain_data):
        '''Write pretrain lookup data to json file'''
        looped = False
        for d in pretrain_data:
            if d['instr_id'] in self.outputs:
                looped = True
                break
            else:
                self.results[d['instr_id']] = {
                    'scan':         d['scan'],
                    'instr_id':     d['instr_id'],
                    'trajectory':   d['trajectory'],  # [(0,0,0), (1,0,0), (0,1,0), ...]
                    'agent_path':   d['agent_path']   # [(viewpoint, viewIndex), ...]
                }
        return looped
        
    def _setup(self, env):
        self.env = env

    def rollout(self, iter_idx=None):
        # Reset environment
        obs, gold_trajs = self.env.reset()
        batch_size = len(obs)

        # Pretrain data
        pretrain_data = [{
            'scan': ob['scan'],
            'instr_id': ob['instr_id'],
            'trajectory': [(0,0,0)],
            'agent_path': [(ob['viewpoint'], ob['viewIndex'])],
        } for ob in obs]

        # Whether agent decides to stop
        ended = np.array([False] * batch_size)
        env_action = [None] * batch_size
        episode_len = max(ob['traj_len'] for ob in obs)

        # t = 0
        for t in range(episode_len):
            assert not ended.all()

            # Environment actions
            for i in range(batch_size):
                if not ended[i]:
                    env_action[i] = gold_trajs[i][t]
                else:
                    env_action[i] = (0,0,0)

            # Take nav action
            obs, _ = self.env.step(env_action)

            # Save trajectory output to pretrain data
            for i, ob in enumerate(obs):
                if not ended[i]:
                    pretrain_data[i]['trajectory'].append(env_action[i])
                    pretrain_data[i]['agent_path'].append((ob['viewpoint'], ob['viewIndex']))

                    if t >= ob['traj_len'] - 1:
                        ended[i] = True
            
        return pretrain_data

    def build_pretrain_data(self, env, n_iters):
        self._setup(env)

        for iter in range(1, n_iters + 1):
            pretrain_data = self.rollout(iter)
            looped = self._collect_pretrain_data(pretrain_data)
            if looped:
                print('Rolled-out {} trajectories.'.format(len(self.results.keys())))
                break
        agent.write_results()


def set_path():

    # Set data load path
    DATA_DIR = os.getenv('PT_DATA_DIR')
    hparams.data_path = os.path.join(DATA_DIR, hparams.data_dir)  #e.g. $PT_DATA_DIR/asknav/
    hparams.img_features = os.path.join(DATA_DIR, hparams.img_features)

    # Set results path to write pretain data to
    hparams.results_path = os.path.join(DATA_DIR, "pretrain/asknav_pretrain_lookup.json")    


if __name__ == '__main__':

    # set data dir
    os.environ['PT_DATA_DIR'] = '/home/hoyeung/blob_matterport3d/'

    # load hparams
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_file', type=str,
        help='configuration file')
    parser.add_argument('-n_iters', type=str,
        help='Number of iterations')
    args = parser.parse_args()

    # Read configuration from a json file
    with open(args.config_file) as f:
        hparams = Namespace(**json.load(f))

    set_path()
    print ("set_path() is done")

    # Setup environment
    tr_env = VNLABuildPretrainBatch(hparams, 'train')

    # Setup agent.results_path
    agent = BuildPretrainDataAgent(hparams)
    agent.build_pretrain_data(tr_env, hparams.n_iters)