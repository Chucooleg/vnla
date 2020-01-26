# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
from collections import defaultdict
import numpy as np
import os


def nested_defaultdict():
    '''Use this rather than lambda to allow defaultdict objects to be pickled'''
    return defaultdict(list)


class HistoryBuffer(object):
    '''
    History Buffer for training value estimation agents
    '''

    def __init__(self, hparams):
        
        # Stored data format
        # (instr_id, global_iter_idx) : {viewpointIndex:[], scan:'', instruction: [], feature: [], action: [], q_values_target: [], ... etc}
        self._indexed_data = defaultdict(nested_defaultdict)

        # Use this to remove earliest experience when buffer is full
        self._earliest_iter_idx = 0

        # Map global_iter_idx to set of instr_id
        self._iter_instr_map = defaultdict(set)

        # Set of (instr_id, global_iter_idx) keys for sampling minibatch
        self._instr_iter_keys = set()

        # max storage limit -- number of examples
        self._curr_buffer_size = 0
        self.max_buffer_size = hparams.max_buffer_size

    def __len__(self):
        return len(self._indexed_data)

    def curr_buffer_size(self):
        return self._curr_buffer_size

    def __getitem__(self, instr_iter_key):
        '''instr_iter_key should be (instr_id, global_iter_idx)'''
        return self._indexed_data[instr_iter_key]

    def save_buffer(self, dir_path):
        '''save history buffer files to a directory)'''
        
        print ('saving history buffer to {}'.format(dir_path))
        
        info_dict = {}
        info_dict['_earliest_iter_idx'] = self._earliest_iter_idx
        info_dict['_iter_instr_map'] = self._iter_instr_map
        info_dict['_instr_iter_keys'] = self._instr_iter_keys
        info_dict['_curr_buffer_size'] = self._curr_buffer_size
        info_dict['max_buffer_size'] = self.max_buffer_size
        
        data_dict = {}
        for key in self._instr_iter_keys:
            data_dict[key] = {
                'viewpointIndex' : self._indexed_data[key]['viewpointIndex'],
                'scan' : self._indexed_data[key]['scan'],
                'instruction' : self._indexed_data[key]['instruction'],
                'feature' : self._indexed_data[key]['feature'],
                'action' : self._indexed_data[key]['action'],
                'view_index_mask_indices' : self._indexed_data[key]['view_index_mask_indices'],
                'viewix_actions_map' : self._indexed_data[key]['viewix_actions_map'],
                'q_values_target' : self._indexed_data[key]['q_values_target']
                }

        info_file_path = os.path.join(dir_path, 'history_buffer_info.pickle')
        data_file_path = os.path.join(dir_path, 'history_buffer_data.pickle') 
        with open(info_file_path, 'wb') as handle:
            pickle.dump(info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(data_file_path, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)	
        
    def load_buffer(self, dir_path):
        '''load history buffer files to object'''
        
        print ('loading history buffer from {}'.format(dir_path))
        
        info_file_path = os.path.join(dir_path, 'history_buffer_info.pickle')
        data_file_path = os.path.join(dir_path, 'history_buffer_data.pickle') 
        with open(info_file_path, 'rb') as handle:
            info_dict = pickle.load(handle)
        with open(data_file_path, 'rb') as handle:
            data_dict = pickle.load(handle)
            
        self._earliest_iter_idx = info_dict['_earliest_iter_idx']
        self._iter_instr_map = info_dict['_iter_instr_map']
        self._instr_iter_keys = info_dict['_instr_iter_keys']
        self._curr_buffer_size = info_dict['_curr_buffer_size']
        self.max_buffer_size = info_dict['max_buffer_size']
            
        for key in self._instr_iter_keys:
            self._indexed_data[key]['viewpointIndex'] = data_dict[key]['viewpointIndex']
            self._indexed_data[key]['scan'] = data_dict[key]['scan']
            self._indexed_data[key]['instruction'] = data_dict[key]['instruction']
            self._indexed_data[key]['feature'] = data_dict[key]['feature']
            self._indexed_data[key]['action'] = data_dict[key]['action']
            self._indexed_data[key]['view_index_mask_indices'] = data_dict[key]['view_index_mask_indices']
            self._indexed_data[key]['viewix_actions_map'] = data_dict[key]['viewix_actions_map']
            self._indexed_data[key]['q_values_target'] = data_dict[key]['q_values_target']

    def remove_earliest_experience(self, batch_size):
        '''Remove earliest iterations of experiences from buffer, until we have room to add the next batch.'''
        while self._curr_buffer_size + batch_size >= self.max_buffer_size:
            print("removing earliest experience")
            remove_size = 0
            for instr_id in self._iter_instr_map[self._earliest_iter_idx]:
                remove_size += len(self._indexed_data[(instr_id, self._earliest_iter_idx)]['action'])
                del self._indexed_data[(instr_id, self._earliest_iter_idx)]
                self._instr_iter_keys.remove((instr_id, self._earliest_iter_idx))
            self._curr_buffer_size -= remove_size
            self._earliest_iter_idx += 1
            print("current buffer size = {}".format(self._curr_buffer_size))
            print("current earliest iter idx = {}".format(self._earliest_iter_idx))

    def is_full(self):
        '''Check if the buffer has stored up to its storage limit'''
        return self._curr_buffer_size >= self.max_buffer_size

    def add_experience(self, experience_batch):
        '''
        Add a single batch of experience to the buffer.
        Called per time step during rollout().
        '''

        batch_size = len(experience_batch)
        if self._curr_buffer_size + batch_size >= self.max_buffer_size:
            self.remove_earliest_experience(batch_size)

        try:
            assert not self.is_full()
        except:
            print("debug")

        # Write experience to buffer
        for experience in experience_batch:
            key = (experience['instr_id'], experience['global_iter_idx'])

            # Append image feature related indices
            self._indexed_data[key]['viewpointIndex'].append(experience['viewpointIndex'])
            self._indexed_data[key]['scan'] = experience['scan']

            # Append list of token IDs
            self._indexed_data[key]['instruction'].append(experience['instruction'])

            # Append tensors
            self._indexed_data[key]['feature'].append(experience['feature'])
            self._indexed_data[key]['action'].append(experience['action'])
            
            # Append pano indexing
            # np array
            self._indexed_data[key]['view_index_mask_indices'].append(experience['view_index_mask_indices'])
            # tensor
            self._indexed_data[key]['viewix_actions_map'].append(experience['viewix_actions_map'])
            
            #  # np array
            # self._indexed_data[key]['end_target_indices'].append(experience['end_target_indices'])

            # Append target
            self._indexed_data[key]['q_values_target'].append(experience['q_values_target'])

            # Build global_iter_idx to instr_id map
            self._iter_instr_map[experience['global_iter_idx']].add(experience['instr_id'])
            # Build set of keys for sampling
            self._instr_iter_keys.add(key)

        self._curr_buffer_size += batch_size

    def sample_minibatch(self, batch_size):
        '''Sample a training batch for training'''

        assert len(self._indexed_data) >= batch_size, \
            '''Buffer doesn't have enough history to sample a batch'''

        # Sample (global_idx, instr_id) key pairs
        # shape (number of unique key pairs, 2)
        instr_key_pairs_list = sorted(list(self._instr_iter_keys))
        # shape (batch_size, 2)
        sampling_indices = np.random.randint(len(instr_key_pairs_list), size=batch_size)
        sampled_iter_instr_key_pair = [instr_key_pairs_list[idx] for idx in sampling_indices]

        # Debug
        # print ('training sampled training keys length', len(sampling_indices))
        # print ('training sampled instr_key_pairs_list', sorted(sampling_indices))

        # Further sample the time step
        traj_lens = [len(self._indexed_data[key]['action']) for key in sampled_iter_instr_key_pair]

        # Debug
        if 1 in traj_lens or 0 in traj_lens:
            print ("debug")

        # Sample timesteps
        # do not sample from t=0 at <start> state
        sampled_timesteps = [np.random.randint(low=1, high=t_len) for t_len in traj_lens]

        # # Vectorized way to choose nearly random timesteps with upperbounds
        # random_ints = np.random.randint(1000000000, size=len(traj_lens))
        # sampled_timesteps = random_ints % traj_lens

        assert len(sampled_iter_instr_key_pair) == len(sampled_timesteps) == batch_size
        return sampled_iter_instr_key_pair, sampled_timesteps
