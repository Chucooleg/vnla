# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pickle
from collections import defaultdict
import numpy as np


class HistoryBuffer(object):
    '''
    History Buffer for training value estimation agents
    '''

    def __init__(self, hparams):
        
        # Stored data format
        # (instr_id, global_iter_idx) : {viewpointIndex:[], scan:'', instruction: [], feature: [], action: [], q_values_target: [], ... etc}
        self._indexed_data = defaultdict(lambda: defaultdict(list))

        # Use this to remove earliest experience when buffer is full
        self._earliest_iter_idx = 0

        # Map global_iter_idx to set of instr_id
        self._iter_instr_map = defaultdict(set)

        # Set of (instr_id, global_iter_idx) keys for sampling minibatch
        self._instr_iter_keys = set()

        # max storage limit -- number of examples
        self._curr_buffer_size = 0
        self._max_buffer_size = hparams.max_buffer_size

    def __len__(self):
        return len(self._indexed_data)

    def __getitem__(self, instr_iter_key):
        '''instr_iter_key should be (instr_id, global_iter_idx)'''
        return self._indexed_data[instr_iter_key]

    def save_buffer(self, file_path):
        '''save buffer to a filepath'''
        # can try to find a better way than pickling...

        with open(file_path, 'wb') as handle:
            pickle.dump({
                '_indexed_data': self._indexed_data,
                '_earliest_iter_idx': self._earliest_iter_idx,
                '_iter_instr_map': self._iter_instr_map,
                '_instr_iter_keys': self._instr_iter_keys,
                '_curr_buffer_size': self._curr_buffer_size,
                '_max_buffer_size': self._max_buffer_size,
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_buffer(self, file_path):
        '''load buffer from a filepath'''

        with open(file_path, 'rb') as handle:
            loaded = pickle.load(handle)

        self._indexed_data = loaded['_indexed_data']
        self._earliest_iter_idx = loaded['_earliest_iter_idx']
        self._iter_instr_map = loaded['_iter_instr_map']
        self._instr_iter_keys = loaded['_instr_iter_keys']
        self._curr_buffer_size = loaded['_curr_buffer_size']
        self._max_buffer_size = loaded['_max_buffer_size']

    def remove_earliest_experience(self):
        '''Remove earliest iteration of experiences from buffer'''
        for instr_id in self._iter_instr_map[self._earliest_iter_idx]:
            del self._indexed_data[(instr_id, self._earliest_iter_idx)]
        self._earliest_iter_idx += 1
        self._curr_buffer_size -= len(self._iter_instr_map[self._earliest_iter_idx])

    def is_full(self):
        '''Check if the buffer has stored up to its storage limit'''
        return self._curr_buffer_size >= self._max_buffer_size

    def add_experience(self, experience_batch):
        '''
        Add a single batch of experience to the buffer.
        Called per time step during rollout().
        '''
        assert not self.is_full()

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

        self._curr_buffer_size += len(experience_batch)

    def sample_minibatch(self, batch_size):
        '''Sample a training batch for training'''

        assert len(self._indexed_data) >= batch_size, \
            '''Buffer doesn't have enough history to sample a batch'''

        # Sample (global_idx, instr_id) key pairs
        # shape (number of unique key pairs, 2)
        instr_key_pairs_list = list(self._instr_iter_keys)
        # shape (batch_size, 2)
        sampling_indices = np.random.randint(len(instr_key_pairs_list), size=batch_size)
        sampled_iter_instr_key_pair = [instr_key_pairs_list[idx] for idx in sampling_indices]

        # Further sample the time step
        traj_lens = [len(self._indexed_data[key]['action']) for key in sampled_iter_instr_key_pair]

        # Sample timesteps
        # do not sample from t=0 at <start> state
        sampled_timesteps = [np.random.randint(low=1, high=t_len) for t_len in traj_lens]

        # # Vectorized way to choose nearly random timesteps with upperbounds
        # random_ints = np.random.randint(1000000000, size=len(traj_lens))
        # sampled_timesteps = random_ints % traj_lens

        assert len(sampled_iter_instr_key_pair) == len(sampled_timesteps) == batch_size
        return sampled_iter_instr_key_pair, sampled_timesteps
