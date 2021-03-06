import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import utils

NAV_ACTIONS =  ['left', 'right', 'up', 'down', 'forward', '<end>']
ASK_ACTIONS = ['dont_ask', 'ask']

# cyan, magenta, yellow
# blue, green, red
# violet, turquoise, orange
# ocean, raspberry, spring green
REF_COLORS = np.array([ '#00FFFF', '#FF00FF', '#FFFF00',
                        '#0000FF', '#00FF00', '#FF0000',
                        '#7700FF', '#00FF77', '#FF7700',
                        '#0077FF', '#FF0077', '#77FF00'] * 10)

class OutputData(object):

    def __init__(self, output_data):
        """
        :param output_data: json object loaded from output/*.json file
        """
        self.output_data = output_data

    @classmethod
    def compute_softmax(cls, logits):
        """arr: shape (num_classes,) from logits."""
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum(axis=0)

    @classmethod
    def compute_entropy(cls, logits):
        """arr: shape (num_classes,) from logits.
        uses natural log base"""
        softmax = cls.compute_softmax(logits)
        return scipy.stats.entropy(softmax)

    @classmethod
    def compute_cross_entropy(cls, logits, gold_idx):
        """arr: shape (num_classes,) from logits.
        uses natural log base"""
        softmax = cls.compute_softmax(logits)
        return - np.log(softmax[gold_idx])

    @classmethod
    def compute_entropy_per_datapt(cls, datapt):
        """For every timestep of a datapt, compute an entropy value based on its predicted logits
        
        Arguments:
            datapt {dictionary} -- a datapt should record info for an entire episode with a unique instr_id`

        Returns:
            3 numpy arrays for navigation inital entropy, navigation final entropy, asking entropy
            each of shape (num timestep,)
        """
        nav_ent_initial_list = [cls.compute_entropy(time_step) for time_step in datapt['agent_nav_logits_initial']]
        nav_ent_final_list = [cls.compute_entropy(time_step) for time_step in datapt['agent_nav_logits_final']]
        ask_ent_list = [cls.compute_entropy(time_step) for time_step in datapt['agent_ask_logits']]
        return nav_ent_initial_list, nav_ent_final_list, ask_ent_list

    @classmethod
    def compute_softmax_per_datapt(cls, datapt):
        """For every timestep of a datapt, compute a softmax vector based on its predicted logits
        
        Arguments:
            datapt {dictionary} -- a datapt should record info for an entire episode with a unique instr_id`

        Returns:
            3 numpy arrays for navigation inital softmax, navigation final softmax, asking softmax
            each of shape (num timestep, num decision classes)
        """
        nav_softmax_initial_list = [cls.compute_softmax(time_step) for time_step in datapt['agent_nav_logits_initial']]
        nav_softmax_final_list = [cls.compute_softmax(time_step) for time_step in datapt['agent_nav_logits_final']]
        ask_softmax_list = [cls.compute_softmax(time_step) for time_step in datapt['agent_ask_logits']]
        return nav_softmax_initial_list, nav_softmax_final_list, ask_softmax_list

    @classmethod
    def compute_cross_entropy_per_datapt(cls, datapt):
        num_decisions = len(datapt['trajectory']) - 1
        nav_cross_ent_initial_list = [
            cls.compute_cross_entropy(datapt['agent_nav_logits_initial'][i], datapt['teacher_nav'][i]) for i in
            range(num_decisions)]
        nav_cross_ent_final_list = [
            cls.compute_cross_entropy(datapt['agent_nav_logits_final'][i], datapt['teacher_nav'][i]) for i in
            range(num_decisions)]
        ask_cross_ent_list = [cls.compute_cross_entropy(datapt['agent_ask_logits'][i], datapt['teacher_ask'][i])
                                    for i in range(num_decisions)]
        return nav_cross_ent_initial_list, nav_cross_ent_final_list, ask_cross_ent_list

    def compute_entropy_output_data(self):
        for i in range(len(self.output_data)):
            nav_ent_initial_list, nav_ent_final_list, ask_ent_list = \
                self.compute_entropy_per_datapt(self.output_data[i])
            self.output_data[i]['agent_nav_ent_initial'] = nav_ent_initial_list
            self.output_data[i]['agent_nav_ent_final'] = nav_ent_final_list
            self.output_data[i]['agent_ask_ent'] = ask_ent_list

    def compute_cross_entropy_output_data(self):
        for i in range(len(self.output_data)):
            nav_cross_ent_initial_list, nav_cross_ent_final_list, ask_cross_ent_list = \
                self.compute_cross_entropy_per_datapt(self.output_data[i])
            self.output_data[i]['agent_nav_cross_ent_initial'] = nav_cross_ent_initial_list
            self.output_data[i]['agent_nav_cross_ent_final'] = nav_cross_ent_final_list
            self.output_data[i]['agent_ask_cross_ent'] = ask_cross_ent_list

    def compute_softmax_output_data(self):
        for i in range(len(self.output_data)):
            nav_softmax_initial_list, nav_softmax_final_list, ask_softmax_list = \
                self.compute_softmax_per_datapt(self.output_data[i])
            self.output_data[i]['agent_nav_softmax_initial'] = nav_softmax_initial_list
            self.output_data[i]['agent_nav_softmax_final'] = nav_softmax_final_list
            self.output_data[i]['agent_ask_softmax'] = ask_softmax_list       

    def compute_nav_argmax_output_data(self):
        for i in range(len(self.output_data)):
            timesteps = len(self.output_data[i]['teacher_ask'])
            agent_nav_argmax_initial = []
            agent_nav_argmax_final = []
            for j in range(timesteps):
                agent_nav_argmax_initial.append(np.argmax(self.output_data[i]['agent_nav_logits_initial'][j]))
                agent_nav_argmax_final.append(np.argmax(self.output_data[i]['agent_nav_logits_final'][j]))
            self.output_data[i]['agent_nav_argmax_initial'] = agent_nav_argmax_initial
            self.output_data[i]['agent_nav_argmax_final'] = agent_nav_argmax_final

    def add_timestep(self):
        for datapt in self.output_data:
            datapt['timesteps'] = [t for t in range(len(datapt['teacher_nav']))]

    def add_subgraph(self):
        # TODO
        pass

    def __getitem__(self, datapt_idx):
        return self.output_data[datapt_idx]


class PlotUtils(object):

    @classmethod
    def _parse_action_type(cls, action_type, cross_ent_bool=False):
        """
        Specify whether we are analyzing navigation or ask actions, to extract matching key strings.
        :param action_type: string. should contain pattern to specific whether nav or ask. initial or final (nav)
        :param cross_ent_bool: (o) boolean. True if we want to look at cross-entropy instead of entropy
        :return: 3 strings that can be used as keys to extract arrays from OutputData[data_pt_i].
                 e.g. OutputData[51]['agent_nav_argmax_final'], OutputData[51]['teacher_nav']
        """
        teacher_target_str, agent_argmax_str, agent_ent_str = "", "", ""
        if 'nav' in action_type:
            teacher_target_str = 'teacher_nav'
            agent_argmax_str = 'agent_nav_argmax'
            agent_ent_str = 'agent_nav_ent'
            agent_softmax_str = 'agent_nav_softmax'
            # for verbal ask agent, the agent has to predict navigation twice
            if 'initial' in action_type:
                agent_argmax_str += "_initial"
                agent_ent_str += "_initial"
                agent_softmax_str += "_initial" 
            elif 'final' in action_type:
                agent_argmax_str += "_final"
                agent_ent_str += "_final"
                agent_softmax_str += "_final"
        elif 'ask' in action_type:
            teacher_target_str = "teacher_ask"
            agent_argmax_str = "agent_ask"
            agent_ent_str = "agent_ask_ent"
            agent_softmax_str = 'agent_ask_softmax'
        # compute cross entropy instead of entropy
        if cross_ent_bool:
            agent_ent_str = agent_ent_str.replace("_ent", "_cross_ent")
        return teacher_target_str, agent_argmax_str, agent_ent_str, agent_softmax_str

    @classmethod
    def flatten_targets_softmaxes_timesteps(cls, data, action_type):
        """
        flatten targets, softmax values and timestep
        :param data: an OutputData data object.
        :param action_type: string. should contain pattern to specific whether nav or ask. initial or final (nav)

        :return: 3 numpy arrays. 
                 teacher_targets_flattened, timesteps_flattened each has shape (sum_datapt(timesteps in datapt i), )
                 agent_softmaxes_flattened has shape (sum_datapt(timesteps in datapt i), num prediction classes)
        """
        # teacher_target_str, agent_argmax_str, agent_ent_str, agent_softmax_str = cls._parse_action_type(action_type, cross_ent_bool)
        teacher_target_str, _, _, agent_softmax_str = cls._parse_action_type(action_type, False)
        teacher_targets_flattened = np.array([target for datapt in data for target in datapt[teacher_target_str]])
        # agent_softmaxes_flattened shape (sum_datapt(timesteps in datapt i), num prediction classes)
        agent_softmaxes_flattened = np.array([softmax_arr for datapt in data for softmax_arr in datapt[agent_softmax_str]])
        timesteps_flattened = np.array([timestep for datapt in data for timestep in datapt['timesteps']])
        #timesteps_flattened = np.array([timestep for datapt in data for timestep in range(len(datapt[teacher_target_str]))])
        return teacher_targets_flattened, agent_softmaxes_flattened, timesteps_flattened

    @classmethod
    def flatten_targets_argmaxes_entropies_timesteps(cls, data, action_type, cross_ent_bool=False):
        """
        flatten targets, argmaxes, entropy values and timestep
        :param data: an OutputData data object.
        :param action_type: string. should contain pattern to specific whether nav or ask. initial or final (nav)
        :param cross_ent_bool: (o) boolean. True if we want to look at cross-entropy instead of entropy
        :return: 4 numpy arrays. each of shape (sum_datapt(timesteps in datapt i), )
        """
        teacher_target_str, agent_argmax_str, agent_ent_str, _ = cls._parse_action_type(action_type, cross_ent_bool)
        teacher_targets_flattened = np.array([target for datapt in data for target in datapt[teacher_target_str]])
        agent_argmaxes_flattened = np.array([argmax for datapt in data for argmax in datapt[agent_argmax_str]])
        agent_entropies_flattened = np.array([entropy for datapt in data for entropy in datapt[agent_ent_str]])
        timesteps_flattened = np.array([timestep for datapt in data for timestep in datapt['timesteps']])
        return teacher_targets_flattened, agent_argmaxes_flattened, agent_entropies_flattened, timesteps_flattened

    @classmethod
    def flatten_locations(cls, data):
        """
        Arguments:
            data {OutputData} -- data object instantiated from OutputData class
        
        Returns:
            viewpt_flattened -- 1-D numpy array of shape (sum_datapt(num timesteps at datapt i), )
            scan_flattened -- same as viewpt_flattened
        """
        step_str = "trajectory"
        scan_str = "scan"
        viewpt_flattened = np.array([traj_step[0] for datapt in data for traj_step in datapt[step_str][:-1]])
        scan_flattened = []
        for datapt in data:
            # number of locations = number of decisions + 1
            len_traj = len(datapt['trajectory']) - 1
            scan_flattened.extend([datapt[scan_str]] * len_traj)
        scan_flattened = np.array(scan_flattened)
        assert viewpt_flattened.shape == scan_flattened.shape
        return viewpt_flattened, scan_flattened

    @classmethod
    def flatten_masks(cls, data, action_type):
        """
        Arguments:
            data {OutputData} -- data object instantiated from OutputData class
        
        Returns:
            masks_flattened -- 1-D numpy array of shape (sum_datapt(num timesteps at datapt i), )
        """           
        if "nav" in action_type:
            mask_str = "agent_nav_logit_masks"
        elif "ask" in action_type:
            mask_str = "agent_ask_logit_masks"
        # check if mask was applied
        check_mask = lambda x: 1 if np.sum(x) > 0 else 0
        masks_flattened = [check_mask(mask) for datapt in data for mask in datapt[mask_str]]
        return np.array(masks_flattened)

    @classmethod
    def shift_and_flatten_vals_by_time_step_delta(cls, output_data, keyname, timestep_delta):
        """For each datapoint in output data, for a key array of interest, such as the array stored under key "agent_nav_ent", create an array with values shifted by `timestep_delta` number of timesteps and an array with curr values but trimmed for alignment's sake.
        
        Arguments:
            output_data {list} -- a list of 1+ OutputData objects. Usually one for test seen and one for test unseen.
            keyname {str} -- name of key with array vals to be time shifted. 
                             e.g. "agent_nav_ent"
            timestep_delta {int} -- cam be positive, negative but not zero. Number of time steps to be shifted for the array values.
        Return:
            Two 1-D numpy arrays each of shape (sum_datapt(num timesteps at datapt i), )
        """
        assert isinstance(timestep_delta, int)
        assert timestep_delta != 0
        curr_flattened = []
        shifted_flattened = []

        # need to time shift within each datapt, not the flattened version
        for data_pt in output_data:
            if timestep_delta > 0:
                # e.g. delta is +1           
                arr_curr = data_pt[keyname][:-timestep_delta] # A B C D -> A B C
                arr_shifted = data_pt[keyname][timestep_delta:] # A B C D -> B C D
            else: 
                # e.g. delta is -1
                arr_curr = data_pt[keyname][-timestep_delta:] # A B C D -> B C D
                arr_shifted = data_pt[keyname][:timestep_delta] # A B C D -> A B C
            curr_flattened.extend(arr_curr)
            shifted_flattened.extend(arr_shifted)
        assert len(curr_flattened) == len(shifted_flattened)
        return curr_flattened, shifted_flattened

    @classmethod
    def get_roomlabels(cls, data):
        """
        Arguments:
            data {OutputData} -- data object instantiated from OutputData class
        
        Returns:
            room_flattened -- 1-D numpy array of shape (sum_datapt(num timesteps at datapt i), )
        """
        viewpt_flattened, scan_flattened = cls.flatten_locations(data=data)
        # panos_to_region[scan][viewpt id] = `<abbreviated roomtag>`
        panos_to_region = {}
        scan_set = set(scan_flattened)
        for scan in scan_set:
            panos_to_region[scan] = utils.load_panos_to_region(scan,"")
        room_flattened = [panos_to_region[scan][viewpt] for scan, viewpt in zip(scan_flattened, viewpt_flattened)]
        return np.array(room_flattened)

    @classmethod
    def _split_ent_data(cls, flattened_data, by_timestep=False):
        """
        split data by tuple keys (teacher gold target, agent predicted argmax target)

        :param flattened_data: a list of the three or four(if by_timestep=True) array elements returned from cls.flatten_targets_argmaxes_entropies_timesteps()

        :param by_timestep: bool. True will further split the data by timestep in the episode.
        """
        if by_timestep:
            assert len(flattened_data) == 4
            teacher_targets_flattened, agent_argmaxes_flattened, agent_entropies_flattened, timesteps_flattened  = \
                flattened_data[0], flattened_data[1], flattened_data[2], flattened_data[3]
        else:
            assert len(flattened_data) == 3
            teacher_targets_flattened, agent_argmaxes_flattened, agent_entropies_flattened  = \
                flattened_data[0], flattened_data[1], flattened_data[2]           
        split_data = defaultdict(list)
        for i in range(len(teacher_targets_flattened)):
            if by_timestep:
                split_data[(teacher_targets_flattened[i], agent_argmaxes_flattened[i], timesteps_flattened[i])].append(agent_entropies_flattened[i])
            else:
                split_data[(teacher_targets_flattened[i], agent_argmaxes_flattened[i])].append(agent_entropies_flattened[i])
        return split_data

    @classmethod
    def _split_softmax_data_by_action_class(cls, flattened_data, action_reference):
        """[summary]
        
        :param flattened_data: a list of the three array elements returned from cls.flatten_targets_softmaxes_timesteps()

        Returns:
            A nested dictionary.
            split_data[forward index]= {'agent_softmax_vals':[...], 'teacher_target_bools': [.....], 'timesteps':[...]}
        """
        assert len(flattened_data) == 3
        teacher_targets_flattened, agent_softmaxes_flattened, timesteps_flattened  = \
            flattened_data[0], flattened_data[1], flattened_data[2]
        
        split_data = defaultdict(lambda : defaultdict(list))
        for i in range(len(timesteps_flattened)):
            for j in range(len(action_reference)):
                split_data[j]['agent_softmax_val'].append(agent_softmaxes_flattened[i][j])
                split_data[j]['teacher_target_bools'].append(int(teacher_targets_flattened[i]) == j)
                split_data[j]['timesteps'].append(timesteps_flattened[i])
        return split_data

    @classmethod
    def _bin_softmax_compute_stats(cls, vals_dict, num_bins):
        """bin using softmax ranges to compute stats for teacher target and agent predicted softmax.
        
        Arguments:
            vals_dict {dict} -- has keys 'agent_softmax_vals', 'teacher_target_bools', 'timesteps'. Each key linking to a 1-D numeric array. e.g.set val_dicts as what `splits["test_seen"][forward action idx=j]` returns

        Returns:
            bin_data {dict} -- has 4 keys. 
                            i.e. {
                                'agent_softmax_avg':[..<num bin vals>..],
                                'agent_softmax_std':[..<num bin vals>..],
                                'teacher_target_avg':[..<num bin vals>..],
                                'teacher_target_std':[..<num bin vals>..],
                                'timesteps' : [[...], [...], [...], [...], ...],
                                'agent_softmax_vals' : [[...], [...], [...], [...], ...],
                                'teacher_target_vals' : [[...], [...], [...], [...], ...]}. 
        """
        agent_softmax_vals = np.array(vals_dict['agent_softmax_val'])
        teacher_target_bools = np.array(vals_dict['teacher_target_bools'])
        timesteps = np.array(vals_dict['timesteps'])

        # set intervals
        bin_width = 1./num_bins
        cuts = list(np.arange(0.0, 1.0, bin_width).round(decimals=2))
        intervals = [(round(cut,2), round(cut + bin_width,2)) for cut in cuts]

        agent_softmax_avg = np.zeros(len(intervals))
        agent_softmax_std = np.zeros(len(intervals))
        teacher_target_avg = np.zeros(len(intervals))
        teacher_target_std = np.zeros(len(intervals))
        timesteps_filtered = [ None for i in range(len(intervals))]
        agent_softmaxes_filtered = [ None for i in range(len(intervals))]
        teacher_targets_filtered = [ None for i in range(len(intervals))]

        for i, interval in enumerate(intervals):
            # filter construction         
            lower = agent_softmax_vals >= interval[0] 
            upper = agent_softmax_vals < interval[1]
            filter = lower & upper
            # filter down to values within the interval
            timesteps_filtered[i] = timesteps[filter]
            agent_softmaxes_filtered[i] = agent_softmax_vals[filter]
            teacher_targets_filtered[i] = teacher_target_bools[filter]
            # compute stats for these filtered values
            agent_softmax_avg[i] = np.mean(agent_softmax_vals[filter])
            agent_softmax_std[i] = np.std(agent_softmax_vals[filter])
            teacher_target_avg[i] = np.mean(teacher_target_bools[filter])
            teacher_target_std[i] = np.std(teacher_target_bools[filter])

        bin_data = dict()
        bin_data['agent_softmax_avg'] = agent_softmax_avg
        bin_data['agent_softmax_std'] = agent_softmax_std
        bin_data['teacher_target_avg'] = teacher_target_avg
        bin_data['teacher_target_std'] = teacher_target_std
        bin_data['timesteps_filtered'] = timesteps_filtered
        bin_data['agent_softmaxes_filtered'] = agent_softmaxes_filtered
        bin_data['teacher_targets_filtered'] = teacher_targets_filtered
        return bin_data

    @classmethod
    def plot_overlapping_histograms(cls, arr_list, label_list, title, xlab, ylab, **kwargs):
        """

        :param arr_list: a list of two or more arrays of shape (num datapts to plot, )
        :param label_list: a list of strings. e.g. ['seen', 'unseen']
        :param title : string
        :param xlab : string
        :param ylab : string
        :param kwargs: a dictionary of plt keyword arguments. e.g. {'bins':30, 'alpha'=0.5...}
        :return:
        """
        for i in range(len(label_list)):
            print ("Count {} = {}".format(label_list[i], len(arr_list[i])))
        plt.figure(figsize=(12,6))
        for i, arr in enumerate(arr_list):
            plt.hist(arr, color=REF_COLORS[i], label=label_list[i], **kwargs)
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend()
        plt.show()

    @classmethod
    def plot_bad_decisions_by_action_type(cls, output_data_list, output_data_labels, 
    action_type, action_reference, timestep_specific=True, room_specific=False, 
    mask_specific=False, cutoff_denom=4):
        """Make one single plot to visualize the fraction of good/bad decision quality such as 
        (wrong, confident) for each category type, such as timestep indices or room labels.
        
        Arguments:
            output_data_list {list} -- a list of 1+ OutputData objects. Usually one for test seen and one for test unseen.
            output_data_labels {list} -- a list of strings naming the output datasets stored in output_data_list. e.g. ["test_seen", "test_unseen"]
            action_type {string} -- Should contain pattern to specific whether `nav` or `ask`. `initial` or `final` in additional to `nav`.
            action_reference {list} -- (o). A single list/array of strings each describing an action by the index.
            timestep_specific {boolean} -- If true, set timestep indices to be category type.
            room_specific {boolean} -- If true, set room labels to be category type.
        """
        print("Action Type : {}".format(action_type))
        if not action_reference:
            if "nav" in action_type:
                action_reference = NAV_ACTIONS
            elif "ask" in action_type:
                action_reference = ASK_ACTIONS
        print("Action Reference : {}".format(action_reference))        

        # compute the entropy cut-off to qualify for being a bad decision entropy
        even_dist_entropy = - np.log(1./len(action_reference)) # - log(p(x))
        # what should count as confidently wrong?
        bad_entropy_cutoff = even_dist_entropy/cutoff_denom # cutoff at first quantile
        
        # get flattened lists
        assert len(output_data_list) == len(output_data_labels)
        flattened = defaultdict(lambda : defaultdict(list))
        for i in range(len(output_data_labels)):
            flattened[output_data_labels[i]]['teacher_targets_flattened'], \
            flattened[output_data_labels[i]]['agent_argmaxes_flattened'], \
            flattened[output_data_labels[i]]['agent_entropies_flattened'], \
            flattened[output_data_labels[i]]['timesteps_flattened'] = \
                cls.flatten_targets_argmaxes_entropies_timesteps(output_data_list[i], action_type)

        # create new agent entropy array based on whether agent was confidently wrong
        # condition 1 : agent is wrong. condition 2 : entropy is below cutoff
        for i in range(len(output_data_labels)):
            wrong = flattened[output_data_labels[i]]['teacher_targets_flattened'] != \
                flattened[output_data_labels[i]]['agent_argmaxes_flattened']
            correct = flattened[output_data_labels[i]]['teacher_targets_flattened'] == \
                flattened[output_data_labels[i]]['agent_argmaxes_flattened']
            confident = flattened[output_data_labels[i]]['agent_entropies_flattened'] <= bad_entropy_cutoff
            not_confident = flattened[output_data_labels[i]]['agent_entropies_flattened'] > bad_entropy_cutoff

            flattened[output_data_labels[i]]['filter_correct_and_confident'] = correct & confident
            flattened[output_data_labels[i]]['filter_correct_and_not_confident'] = correct & not_confident
            flattened[output_data_labels[i]]['filter_wrong_and_not_confident'] = wrong & not_confident
            flattened[output_data_labels[i]]['filter_wrong_and_confident'] = wrong & confident
            
        # add flattened room label list for each output dataset (e.g. "r", "-", "h", "...")
        if room_specific:
            for i in range(len(output_data_labels)):
                flattened[output_data_labels[i]]['room_labels_flattened'] = cls.get_roomlabels(output_data_list[i])

        # add flattened mask on/off list for each output dataset
        if mask_specific:
             for i in range(len(output_data_labels)):
                flattened[output_data_labels[i]]['masks_flattened'] = cls.flatten_masks(output_data_list[i], action_type) 

        # split data by timestep/location
        # split['test_unseen'][3] should give a 1-D numeric array of entropy values for time step 3
        # counts_<filter name>['test_unseen'][3] should give a tuple (qualified counts, unqualified counts) for time step 3
        # categories is a set. {1, 2, ... max time step} or {"-", "h". "r", ...}
        ##splits = {}
        counts_correct_confident = {}
        counts_correct_not_confident = {}
        counts_wrong_not_confident = {}
        counts_wrong_confident = {}
   
        categories = None
        for i in range(len(output_data_labels)):
            agent_entropies_flattened = flattened[output_data_labels[i]]['agent_entropies_flattened']
            if timestep_specific:
                categories_flattened = flattened[output_data_labels[i]]['timesteps_flattened']
            elif room_specific:
                categories_flattened = flattened[output_data_labels[i]]['room_labels_flattened']
            elif mask_specific:
                categories_flattened = flattened[output_data_labels[i]]['masks_flattened']

            _, count_data_correct_confident = \
                cls._filter_and_split_ent_by_decision_quality(
                    flattened_data=[agent_entropies_flattened, categories_flattened], 
                    filter=flattened[output_data_labels[i]]['filter_correct_and_confident'])
            _, count_data_correct_not_confident = \
                cls._filter_and_split_ent_by_decision_quality(
                    flattened_data=[agent_entropies_flattened, categories_flattened], 
                    filter=flattened[output_data_labels[i]]['filter_correct_and_not_confident'])
            _, count_data_wrong_not_confident = \
                cls._filter_and_split_ent_by_decision_quality(
                    flattened_data=[agent_entropies_flattened, categories_flattened], 
                    filter=flattened[output_data_labels[i]]['filter_wrong_and_not_confident'])
            _, count_data_wrong_confident = \
                cls._filter_and_split_ent_by_decision_quality(
                    flattened_data=[agent_entropies_flattened, categories_flattened], 
                    filter=flattened[output_data_labels[i]]['filter_wrong_and_confident'])

            counts_correct_confident[output_data_labels[i]] = count_data_correct_confident
            counts_correct_not_confident[output_data_labels[i]] = count_data_correct_not_confident
            counts_wrong_not_confident[output_data_labels[i]] = count_data_wrong_not_confident
            counts_wrong_confident[output_data_labels[i]] = count_data_wrong_confident
            
            # we want to narrow down to the common set of categories between the different output datasets
            if categories is None:
                categories = set(count_data_wrong_confident.keys())
            else:
                categories = categories & set(count_data_wrong_confident.keys())

        # plot grouped bar chart below
        # order the categories first
        category_ids = sorted(list(categories)) # TODO deal too many room types
        if timestep_specific:
            category_name = "timestep"
        elif room_specific:
            category_name = "room tag"
        elif mask_specific:
            category_name = "mask applied"

        # wrong, confidently in test seen; wrong, confidently in test unseen
        arr_a = [[counts_wrong_confident[output_data_labels[i]][cat][0] for cat in category_ids] \
            for i in range(len(output_data_labels))]
        arr_a = np.vstack((arr_a[0], arr_a[1]))

        # wrong, not confidently in test seen; wrong, not confidently in test unseen
        arr_b = np.array([[counts_wrong_not_confident[output_data_labels[i]][cat][0] for cat in category_ids] \
            for i in range(len(output_data_labels))])
        arr_b = np.vstack((arr_b[0], arr_b[1]))

        # correct, not confidently in test seen; wrong, correct, not confidently  in test unseen
        arr_c = np.array([[counts_correct_not_confident[output_data_labels[i]][cat][0] for cat in category_ids] \
            for i in range(len(output_data_labels))])
        arr_c = np.vstack((arr_c[0], arr_c[1]))  

        # correct, confidently in test seen; wrong, correct, confidently  in test unseen
        arr_d = np.array([[counts_correct_confident[output_data_labels[i]][cat][0] for cat in category_ids] \
            for i in range(len(output_data_labels))])
        arr_d = np.vstack((arr_d[0], arr_d[1])) 

        assert len(arr_a) == len(output_data_labels)
        # plot normalized bar chart, plot raw count bar chart 
        for norm in (True, False):
            cls.plot_grouped_bar_comparison(
                category_ids=category_ids,
                arr_a=arr_a, 
                arr_b=arr_b,
                arr_c=arr_c,
                arr_d=arr_d,
                category_name=category_name,
                quality_labels = ['incorrect, confident', 'incorrect, not confident', 'correct, not confident', 'correct, confident'],
                output_data_labels=output_data_labels,
                action_type=action_type,
                figsize=(20, 12), normalized=norm,
                h_line=True, dataset_colors=None)

    @classmethod
    def _filter_and_split_ent_by_decision_quality(cls, flattened_data, filter):
        """Filter down to entropy values by decision quality. 
        Then split by category (e.g. time/location)

        Arguments:
            flattened_data {list} -- a list containing two arrays. 
                                     First entropy value array has shape (sum_data_pt(timesteps datapt i), ). 
                                     Second category (time or room type) has the same shape.
            filter {array} -- shape (sum_data_pt(timesteps datapt i), ). An array of 1s and 0s 
                              indicating the quality of decisions
        
        Returns:
            split_data {defaultdict} -- keys as category indices. values as an array of entropy values
            count_data {defaultdict} -- keys as category indices. values as counts_wrong_confident of (qualified datapts, unqualified datapts)
        """
        # categories such as different time steps, or locations
        agent_entropies_flattened, categories_flattened =  \
            flattened_data[0], flattened_data[1]
        qualified_entropies = agent_entropies_flattened[filter]
        qualified_categories = categories_flattened[filter]

        split_data = defaultdict(list)
        count_data = defaultdict(tuple)

        # split_data[time step 3] = [ent val 1, ent val 2, ent val 3,...]
        for i in range(len(qualified_entropies)):
            split_data[qualified_categories[i]].append(qualified_entropies[i])

        # count_data[time step 3] = (bad confident decisions made in timestep3, total decisions made in timestep3)
        for key in split_data.keys():
            category_total = np.sum(categories_flattened == key)
            qualified_count = len(split_data[key])
            count_data[key] = (qualified_count, category_total - qualified_count)

        return split_data, count_data

    @classmethod
    def plot_time_entropy_corr_by_action_type(cls, output_data_list, output_data_labels, 
    action_type, action_reference, timestep_delta, cutoff_denom=4):

        print("Action Type : {}".format(action_type))
        if not action_reference:
            if "nav" in action_type:
                action_reference = NAV_ACTIONS
            elif "ask" in action_type:
                action_reference = ASK_ACTIONS
        print("Action Reference : {}".format(action_reference))         
        print("Time shift = {}".format(timestep_delta))

        # shift time step per traj and flatten vals
        assert len(output_data_list) == len(output_data_labels)
        teacher_target_str, agent_argmax_str, agent_ent_str, _ = cls._parse_action_type(action_type)
        flattened = defaultdict(lambda : defaultdict(list))
        # flattened['test_seen']['teacher_targets_curr_flattened'] = 1-D array
        for i in range(len(output_data_labels)):
            flattened[output_data_labels[i]]["teacher_targets" + '_curr_flattened'], flattened[output_data_labels[i]]["teacher_targets" + '_shifted_flattened'] = \
                cls.shift_and_flatten_vals_by_time_step_delta(output_data=output_data_list[i], keyname=teacher_target_str, timestep_delta=timestep_delta)
            flattened[output_data_labels[i]]["agent_argmaxes" + '_curr_flattened'], flattened[output_data_labels[i]]["agent_argmaxes" + '_shifted_flattened'] = \
                cls.shift_and_flatten_vals_by_time_step_delta(output_data=output_data_list[i], keyname=agent_argmax_str, timestep_delta=timestep_delta)
            flattened[output_data_labels[i]]["agent_entropies" + '_curr_flattened'], flattened[output_data_labels[i]]["agent_entropies" + '_shifted_flattened'] = \
                cls.shift_and_flatten_vals_by_time_step_delta(output_data=output_data_list[i], keyname=agent_ent_str, timestep_delta=timestep_delta)
            flattened[output_data_labels[i]]["timesteps" + '_curr_flattened'], flattened[output_data_labels[i]]["timesteps" + '_shifted_flattened'] = \
                cls.shift_and_flatten_vals_by_time_step_delta(output_data=output_data_list[i], keyname="timesteps", timestep_delta=timestep_delta)
        
        #  compute filters & plot
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 18))
        # compute the entropy cut-off to qualify for being a bad decision entropy
        even_dist_entropy = - np.log(1./len(action_reference)) # - log(p(x))
        # what should count as confidently wrong?
        bad_entropy_cutoff = even_dist_entropy/cutoff_denom # cutoff at first quantile

        # loop through (correct, correct), (correct, incorrect), (incorrect, correct), (incorrect, incorrect)
        for typ, ax_id in zip([("correct", "correct"), ("correct", "incorrect"), ("incorrect", "correct"), ("incorrect", "incorrect")],
        [(0, 0), (0, 1), (1, 1), (1, 0)]):

             # plot vertical lines
            axes[ax_id].vlines(x=np.arange(0.0, even_dist_entropy + 0.1, 0.1), ymin=0.0,ymax=even_dist_entropy, ls="--", colors="#888888", linewidth=0.1)
             
            # plot diagonal
            axes[ax_id].plot((0, even_dist_entropy), (0, even_dist_entropy), ls="--", c=".3")
            axes[ax_id].hlines(y=even_dist_entropy / 4., xmin=0.0, xmax=even_dist_entropy, ls="--")          
            axes[ax_id].vlines(x=even_dist_entropy / 4., ymin=0.0, ymax=even_dist_entropy, ls="--")   

            # loop thtough test seen and test unseen output data
            for j in range(len(output_data_labels)):
                # curr
                if typ[0] == "correct":
                    filter_curr = np.array(flattened[output_data_labels[j]]['teacher_targets_curr_flattened']) == \
                        np.array(flattened[output_data_labels[j]]['agent_argmaxes_curr_flattened'])
                elif typ[0] == 'incorrect':
                    filter_curr = np.array(flattened[output_data_labels[j]]['teacher_targets_curr_flattened']) != \
                        np.array(flattened[output_data_labels[j]]['agent_argmaxes_curr_flattened'])      
                if typ[1] == "correct":                                     
                    filter_shifted = np.array(flattened[output_data_labels[j]]['teacher_targets_shifted_flattened']) == \
                        np.array(flattened[output_data_labels[j]]['agent_argmaxes_shifted_flattened'])
                elif typ[1] == 'incorrect':
                    filter_shifted = np.array(flattened[output_data_labels[j]]['teacher_targets_shifted_flattened']) != \
                        np.array(flattened[output_data_labels[j]]['agent_argmaxes_shifted_flattened']) 
                fil = filter_curr & filter_shifted

                x = np.array(flattened[output_data_labels[j]]['agent_entropies_curr_flattened'])[fil]
                y = np.array(flattened[output_data_labels[j]]['agent_entropies_shifted_flattened'])[fil]
                axes[ax_id].scatter(x, y, marker='x', c=REF_COLORS[j], label="{} : {}".format(output_data_labels[j], len(x)), alpha=0.15)

                conf_conf = np.sum((x <= bad_entropy_cutoff) & (y <= bad_entropy_cutoff))
                conf_notconf = np.sum((x <= bad_entropy_cutoff) & (y > bad_entropy_cutoff))
                notconf_conf = np.sum((x > bad_entropy_cutoff) & (y <= bad_entropy_cutoff))
                notconf_notconf = np.sum((x > bad_entropy_cutoff) & (y > bad_entropy_cutoff))               

                # Annotate directly on graph
                # split the data by confidence first
                axes[ax_id].annotate("{}:\n{} ({})".format(output_data_labels[j], conf_conf, round(conf_conf * 1. / len(x), 2)), 
                xy=(0 + 0.1, bad_entropy_cutoff - 0.2 - 0.15 * j), fontsize="large")
                axes[ax_id].annotate("{}:\n{} ({})".format(output_data_labels[j], conf_notconf, round(conf_notconf * 1. / len(x), 2)), 
                xy=(0 + 0.1, bad_entropy_cutoff + 0.5 - 0.15 * j), fontsize="large")
                axes[ax_id].annotate("{}:\n{} ({})".format(output_data_labels[j], notconf_conf, round(notconf_conf * 1. / len(x), 2)), 
                xy=(bad_entropy_cutoff + 0.1, bad_entropy_cutoff - 0.2 - 0.15 * j), fontsize="large")
                axes[ax_id].annotate("{}:\n{} ({})".format(output_data_labels[j], notconf_notconf, round(notconf_notconf * 1. / len(x), 2)), 
                xy=(bad_entropy_cutoff + 0.1, bad_entropy_cutoff + 0.5 - 0.15 * j), fontsize="large")

            axes[ax_id].set_title(str(typ))
            axes[ax_id].set_xlabel("Time step `t` Entropy -- {}".format(typ[0].upper()))
            axes[ax_id].set_ylabel("Time step `t {}{}` Entropy -- {}".format("+" if timestep_delta>0 else "", timestep_delta, typ[1].upper()))
            axes[ax_id].xaxis.set_ticks(np.arange(0.0, even_dist_entropy, 0.1))
            axes[ax_id].yaxis.set_ticks(np.arange(0.0, even_dist_entropy, 0.1))
            axes[ax_id].legend(markerscale=1.)

        fig.suptitle(action_type)

    @classmethod
    def plot_entropy_by_action_type(cls, output_data_list, output_data_labels, action_type, cross_ent_bool=False, 
    action_reference=None, action_pair_specific=True, timestep_specific=False, bins=30):
        """
        Plot and visualize all histograms related to an action type to observe agent entropy distribution.
        One histogram is plot for very (teacher gold target, agent predicted argmax) combination.
        :param output_data_list: a list of 1+ OutputData objects. Usually one for test seen and one for test unseen.
        :param output_data_labels: a list of strings naming the output datasets stored in output_data_list. e.g. ["test_seen", "test_unseen"]
        :param action_type: string. Should contain pattern to specific whether `nav` or `ask`. `initial` or `final` in additional to `nav`.
        :param cross_ent_bool: (o) boolean. True if we want to look at cross-entropy instead of entropy
        :param action_reference: (o). A single list/array of strings each describing an action by the index.
                                 e.g. ['left', 'right', 'up', 'down', 'forward'] for nav action type
                                 e.g. ["dont_ask", "ask"] for ask action type
        :param timestep_specific: (o). If True, produce extra plots that split by time step.
        """
        print("Action Type : {}".format(action_type))
        if not action_reference:
            if "nav" in action_type:
                action_reference = NAV_ACTIONS
            elif "ask" in action_type:
                action_reference = ASK_ACTIONS
        print("Action Reference : {}".format(action_reference))

        # flattened data into arrays of shape (sum_datapt(num time steps at datapt i), )
        # flattened['test_unseen']['teacher_targets_flattened'] is an array
        assert len(output_data_list) == len(output_data_labels)
        flattened = defaultdict(lambda: defaultdict(list))
        for i in range(len(output_data_labels)):
            flattened[output_data_labels[i]]['teacher_targets_flattened'], \
            flattened[output_data_labels[i]]['agent_argmaxes_flattened'], \
            flattened[output_data_labels[i]]['agent_entropies_flattened'], \
            flattened[output_data_labels[i]]['timesteps_flattened'] = \
                cls.flatten_targets_argmaxes_entropies_timesteps(output_data_list[i], action_type, cross_ent_bool)

        # plot overall entropy histogram (all action targets)
        kwargs = dict(alpha=0.5, bins=bins, density=True, stacked=True)
        cls.plot_overlapping_histograms(arr_list=[flattened[label]['agent_entropies_flattened'] for label in flattened],
                                        label_list=output_data_labels, # e.g. ['test_seen', 'test_unseen'],
                                        title="{} \n {}Entropy Distribution".format(action_type, "Cross " if cross_ent_bool else ""),
                                        xlab="Per agent decision entropy",
                                        ylab="Density",
                                        **kwargs)

        if action_pair_specific:
            # split data by key (teacher gold target, agent predicted argmax)
            # splits['test_unseen'][(teacher target=1, agent argmax=1)] should give an array
            splits = {}
            for i in range(len(output_data_labels)):
                splits[output_data_labels[i]] = cls._split_ent_data(flattened_data=
                    [flattened[output_data_labels[i]]['teacher_targets_flattened'],
                    flattened[output_data_labels[i]]['agent_argmaxes_flattened'],
                    flattened[output_data_labels[i]]['agent_entropies_flattened']
                    ], by_timestep=False)

            # plot entropy histograms by (teacher target key, agent argmax key)
            actions = range(len(action_reference))
            for teacher_tar_key in actions:
                for agent_argmax_key in actions:
                    cls.plot_overlapping_histograms(arr_list=[splits[label][(teacher_tar_key, agent_argmax_key)] for label in splits],
                                                    label_list=output_data_labels,  # e.g. ['test_seen', 'test_unseen'],
                                                    title='{} \nTeacher {}({}), Agent {}({}), {}Entropy Distribution'.format(
                                                        action_type,
                                                        teacher_tar_key, action_reference[teacher_tar_key],
                                                        agent_argmax_key, action_reference[agent_argmax_key],
                                                        "Cross " if cross_ent_bool else ""),
                                                    xlab="Per agent decision entropy",
                                                    ylab="Density",
                                                    **kwargs)

        if timestep_specific:
            # split data by key (teacher gold target, agent predicted argmax, time steps)
            # splits['test_unseen'][(teacher target=1, agent argmax=1, time steps=5)] should give an array
            splits = {}
            for i in range(len(output_data_labels)):
                splits[output_data_labels[i]] = cls._split_ent_data(flattened_data= [
                    flattened[output_data_labels[i]]['teacher_targets_flattened'],
                    flattened[output_data_labels[i]]['agent_argmaxes_flattened'],
                    flattened[output_data_labels[i]]['agent_entropies_flattened'],
                    flattened[output_data_labels[i]]['timesteps_flattened']
                    ], by_timestep=True)

            # plot entropy histograms by (teacher target keym agent argmax key, timestep key)
            actions = range(len(action_reference))
            max_timestep = max(max(flattened[label]['timesteps_flattened']) for label in output_data_labels)
            for teacher_tar_key in actions:
                for agent_argmax_key in actions:
                    for timestep_key in range(max_timestep):
                        cls.plot_overlapping_histograms(arr_list=[splits[label][(teacher_tar_key, agent_argmax_key, timestep_key)] for label in splits],
                                                        label_list=output_data_labels,  # e.g. ['test_seen', 'test_unseen'],
                                                        title='Teacher {}({}), Agent {}({}), Timestep {}, {}Entropy Distribution'.format(
                                                            teacher_tar_key, action_reference[teacher_tar_key],
                                                            agent_argmax_key, action_reference[agent_argmax_key],
                                                            timestep_key,
                                                            "Cross " if cross_ent_bool else ""),
                                                        xlab="Per agent decision entropy",
                                                        ylab="Density",
                                                        **kwargs)

    @classmethod
    def plot_calibration_graph_by_action_type(cls, output_data_list, output_data_labels, action_type, 
    action_reference=None, time_step_specific=False, num_bins=10):
        """Plot and visualize a calibration graph for each action.
        i.e. one graph for "forward", one grahp for "right",... if action type is nav
        i.e. one graph for "dont_ask, one grahp for "ask",... if action type is ask
        
        Arguments:
            output_data_list {list} -- a list of 1+ OutputData objects. Usually one for test seen and one for test unseen.
            output_data_labels {list} -- a list of strings naming the output datasets stored in output_data_list. e.g. ["test_seen", "test_unseen"]
            action_type {string} -- Should contain pattern to specific whether `nav` or `ask`. `initial` or `final` in additional to `nav`.
            num_bins {int} -- Indicate how fine we want to split softmax prob from 0.0-1.0 into calibration bins.
        
        Keyword Arguments:
            action_reference {list} -- A single list/array of strings each describing an action by the index.
            time_step_specific {bool} -- If true, overlay predictions split by time step on top of graph (default: {False})
        """
        print("Action Type : {}".format(action_type))
        if not action_reference:
            if "nav" in action_type:
                action_reference = NAV_ACTIONS
            elif "ask" in action_type:
                action_reference = ASK_ACTIONS
        print("Action Reference : {}".format(action_reference))   
        
        # flattened data into arrays of shape (sum_datapt(num time steps at datapt i), )
        # flattened['test_unseen']['teacher_targets_flattened'] is a 1-D numeric array
        # flattened['test_unseen']['agent_softmaxes_flattened'] is 2-D numeric array, with shape (sum_datapt(num time steps at datapt i), num prediction classes)
        assert len(output_data_list) == len(output_data_labels)
        flattened = defaultdict(lambda : defaultdict(list))
        for i in range(len(output_data_labels)):
            # shape (sum_datapt(timesteps in datapt i), )
            # shape (sum_datapt(timesteps in datapt i), )
            # shape (sum_datapt(timesteps in datapt i), num prediction classes)
            flattened[output_data_labels[i]]['teacher_targets_flattened'], flattened[output_data_labels[i]]['agent_softmaxes_flattened'], flattened[output_data_labels[i]]['timesteps_flattened'] = cls.flatten_targets_softmaxes_timesteps(output_data_list[i], action_type)

        # splits['test_seen'][forward idx=0] = {'agent_softmax_vals':[...], 'teacher_target_bools': [.....], 'timesteps':[...]}
        splits = {}
        for i in range(len(output_data_labels)):
            splits[output_data_labels[i]] = cls._split_softmax_data_by_action_class(
                flattened_data = [
                    flattened[output_data_labels[i]]['teacher_targets_flattened'],
                    flattened[output_data_labels[i]]['agent_softmaxes_flattened'],
                    flattened[output_data_labels[i]]['timesteps_flattened']
                ],
                action_reference=action_reference
            )

        # binning, compute average and compute variance within bin
        # binned['test_seen'][forward idx=0] = {
                                                # 'agent_softmax_avg':[..<num bin vals>..],
                                                # 'agent_softmax_std':[..<num bin vals>..],
                                                # 'teacher_target_avg':[..<num bin vals>..],
                                                # 'teacher_target_std':[..<num bin vals>..],
                                                # 'timesteps' : [[...], [...], [...], [...], ...],
                                                # 'agent_softmax_vals' : [[...], [...], [...], [...], ...],
                                                # 'teacher_target_vals' : [[...], [...], [...], [...], ...]}
        binned = defaultdict(lambda : defaultdict(dict))
        for i in range(len(output_data_labels)):
            for j in range(len(action_reference)):
                binned[output_data_labels[i]][j] = cls._bin_softmax_compute_stats(splits[output_data_labels[i]][j], num_bins)

        # plot one graph per action class!
        for a in range(len(action_reference)):
            f, ax = plt.subplots(figsize=(12, 12))
            # plot vertical lines
            bin_width = 1./num_bins
            center = np.arange(bin_width / 2, 1 + bin_width / 2, bin_width)
            for i in np.append((center - 1./num_bins/2), 1.0):
                plt.axvline(x=i, ls="--", c="0.5", linewidth=0.1)
            # plot diagonal
            ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
            # plot teacher gold mean
            dotsize_collect = []
            for k in range(len(output_data_labels)):
                y = binned[output_data_labels[k]][a]['teacher_target_avg']
                x = center
                dotsize = [bin_arr.shape[0]*0.75 for bin_arr in binned[output_data_labels[k]][a]['teacher_targets_filtered']]
                dotsize_collect.append(dotsize)
                ax.scatter(x, y, s=dotsize, label=output_data_labels[k] + " teacher mean", alpha=0.75, c=REF_COLORS[k])
            ax.set_title("{}, Action = {}".format(action_type, action_reference[a]))
            ax.xaxis.set_ticks(np.arange(0.0, 1.0 + bin_width*2, bin_width*4))
            ax.legend(markerscale=0.1)

        return dotsize_collect

    @classmethod
    def plot_grouped_bar_comparison(cls, category_ids, arr_a, arr_b, arr_c, arr_d, category_name,
                                    quality_labels, output_data_labels, action_type,
                                    figsize=(20, 12), normalized=True,
                                    h_line=True, dataset_colors=None, ):
        """
        Compare a set test seen and test unseen outputs by quality of agent decisions.
        Using grouped bar chart.
        :param category_ids: numpy array or list shape (num_categories,).
                         e.g. [0, 1, 2, 3] for 4 timesteps
                         e.g. ['h', 'r', '-', ...] for room tags
        :param arr_a: numpy array (num output datasets, num categories). 
                      e.g. (incorrect, confidently in test seen; incorrect, confidently in test unseen)
        :param arr_b: numpy array (num output datasets, num categories).
                      e.g. (incorrect, not confidently in test seen; incorrect, not confidently in test unseen)
        :param arr_c: numpy array (num output datasets, num categories).
                      e.g. (correct, not confidently in test seen, correct, not confidently in test unseen)
        :param arr_d: numpy array (num output datasets, num categories). To add a dotted line mark only
                      e.g. (correct, confidently in test seen, correct, confidently in test unseen)
        :param category_name: string to label x-axis. Such as "time" or "room tag"
        :param quality_labels: 3-string iterable. e.g. ('incorrect, confidently', 'incorrect, not-confidently', 'correct')
        :param output_data_labels: multi-string tuple. e.g. ('test_seen', 'test_unseen')
        :param figsize: numeric tuple.
        :param normalized: boolean. Normalize counts_wrong_confident.
        :param h_line: boolean. Add hline for baseline experiment
        :param dataset_colors: array of strings. hexcode colors.
        :return:
        """
        assert (arr_a.shape == arr_b.shape == arr_c.shape == arr_d.shape and
                arr_a.shape[0] == len(output_data_labels) and
                arr_a.shape[1] == len(category_ids))
        if dataset_colors is not None:
            assert(len(dataset_colors) >= len(output_data_labels))

        rs = [np.arange(len(arr_a[0]))]
        bar_width = min(0.2, 0.5 / arr_a.shape[0])

        # b & w
        colors = ['#000000', '#888888', '#dddddd', '#ffffff']
        if dataset_colors is None:
            dataset_colors = REF_COLORS
        dataset_colors = np.array(dataset_colors)

        if normalized:
            tot = arr_a + arr_b + arr_c + arr_d
            arr_a = np.around(arr_a / tot, 4) 
            arr_b = np.around(arr_b / tot, 4)
            arr_c = np.around(arr_c / tot, 4)
            arr_d = 1 - arr_a - arr_b - arr_c

        # figure boundary
        plt.subplots(figsize=figsize)
        if normalized:
            limit = 1.05
        else:
            limit = np.max(arr_a + arr_b + arr_c + arr_d) * 1.05
        plt.ylim(0, limit)

        p_exp = []
        p1, p2, p3 = None, None, None

        for i in range(arr_a.shape[0]):  # per experiment
            # backdrop to differentiate experiments
            p0 = plt.bar(rs[i], np.repeat(limit + 500, arr_a.shape[1]),
                        width=bar_width + bar_width / float(1.5),
                        edgecolor=dataset_colors[i], label=output_data_labels[i],
                        color=dataset_colors[i], alpha=0.2)
            p_exp.append(p0[0])  # for legends
            # actual data
            p1 = plt.bar(rs[i], arr_a[i], width=bar_width,
                         label=output_data_labels[i],
                         color=colors[0])

            p2 = plt.bar(rs[i], arr_b[i], width=bar_width,
                         label=output_data_labels[i],
                         bottom=arr_a[i], color=colors[1])

            p3 = plt.bar(rs[i], arr_c[i], width=bar_width,
                         label=output_data_labels[i],
                         bottom=arr_a[i] + arr_b[i], color=colors[2])

            p4 = plt.bar(rs[i], arr_d[i], width=bar_width,
                         label=output_data_labels[i],
                         bottom=arr_a[i] + arr_b[i] + arr_c[i], color=colors[3])

            hoffset = bar_width + bar_width / float(1.5)

            # h line
            #if i == 0 and h_line:
            if h_line:
                hspan = hoffset * len(output_data_labels)
                h = plt.hlines(y=arr_a[i] + arr_b[i], xmin=rs[i] - 0.2,
                               # xmax=rs[i] + hspan,
                               xmax=rs[i] + bar_width + 0.1,
                               colors='r', alpha=1.0, linewidth=2.0,
                               linestyles='--', label='baseline')
            # horizontal shift to next trial
            rs.append(rs[i] + hoffset)

        # title
        plt.title("{} \n {} : {} \n {} - {} split".format(
            action_type,
            " vs ".join(output_data_labels),
            str(('normalized' if normalized else 'raw counts')),
            quality_labels[0], quality_labels[1]),
            fontweight='bold', fontsize=20
        )

        plt.xlabel(category_name, fontweight='bold',
                   fontsize=14)
        plt.ylabel('normalized cts' if normalized else 'data point cts',
                   fontweight='bold',
                   fontsize=14)
        plt.xticks([r + bar_width for r in range(arr_a.shape[1] + 1)],
                   category_ids, fontsize=14)

        # legends
        legend1 = plt.legend(p_exp, output_data_labels, fontsize=14,
                             loc='upper right')
        plt.legend((p4[0], p3[0], p2[0], p1[0]), quality_labels[::-1], fontsize=14,
                   loc='upper left')
        plt.gca().add_artist(legend1)
        plt.show()

    def plot_confusion_matrix(y_true, y_pred, classes,
                            normalize=False,
                            title=None,
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        copied from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')
        
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

