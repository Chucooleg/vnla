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
        ask_ent_final_list = [cls.compute_entropy(time_step) for time_step in datapt['agent_ask_logits']]
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
            self.output_data[i]['agent_nav_softmax_initial'] = nav_cross_ent_initial_list
            self.output_data[i]['agent_nav_softmax_final'] = nav_cross_ent_final_list
            self.output_data[i]['agent_ask_softmax_ent'] = ask_cross_ent_list       

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
            elif 'final' in action_type:
                agent_argmax_str += "_final"
                agent_ent_str += "_final"
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
        teacher_target_str, _, _, agent_softmax_str = cls._parse_action_type(action_type, cross_ent_bool)
        teacher_targets_flattened = np.array([target for datapt in data for target in datapt[teacher_target_str]])
        # agent_softmaxes_flattened shape (sum_datapt(timesteps in datapt i), num prediction classes)
        agent_softmaxes_flattened = np.array([softmax_arr for datapt in data for softmax_arr in datapt[agent_softmax_str]])
        timesteps_flattened = np.array([timestep for datapt in data for timestep in range(len(datapt[teacher_target_str]))])
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
        timesteps_flattened = np.array([timestep for datapt in data for timestep in range(len(datapt[teacher_target_str]))])
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
            split_data[forward index]= {'agent_softmax_vals':[...], 'teacher_target_bools': [.....], 'time_steps':[...]}
        """
        assert len(flattened_data) == 3
        teacher_targets_flattened, agent_softmaxes_flattened, 
        timesteps_flattened  = \
            flattened_data[0], flattened_data[1], flattened_data[2]
        
        # TODO double check
        split_data = defaultdict(lambda : defaultdict(list))
        for i in range(len(timesteps_flattened)):
            for j in range(len(action_reference)):
                split_data[j]['agent_softmax_val'].append(agent_softmaxes_flattened[i][j])
                split_data[j]['teacher_target_bools'].append(int(teacher_targets_flattened[i]) == j)
                split_data[j]['time_steps'].append(timesteps_flattened[i])
        return split_data

    @classmethod
    def _bin_softmax_compute_stats(cls, vals_dict):
        """bin using softmax ranges to compute stats for teacher target and agent predicted softmax.
        
        Arguments:
            vals_dict {dict} -- has 3 keys. 'agent_softmax_vals', 'teacher_target_bools' and 'time_steps'. Each key linking to a 1-D numeric array with shape (sum_datapt(num timesteps at datapt i), ). Note this is specific to both output dataset and action class. e.g. from `splits["test_seen"][forward action idx=j]`

        Returns:
            bin_data {dict} -- has 3 keys. i.e. {'agent_softmax_avg':[..<num bin vals>..], 'agent_softmax_std':[..<num bin vals>..], 'teacher_target_avg':[..<num bin vals>..], 'teacher_target_std':[..<num bin vals>..]}. 
        """
        # HERE



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
            plt.hist(arr, color=dataset_colors[i], label=label_list[i], **kwargs)
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend()
        plt.show()

    @classmethod
    def plot_bad_decisions_by_action_type(cls, output_data_list, output_data_labels, 
    action_type, action_reference, timestep_specific, room_specific, cutoff_denom=4):
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
            wrong = flattened[output_data_labels[i]]['teacher_targets_flattened'] == \
                flattened[output_data_labels[i]]['agent_argmaxes_flattened']
            confident = flattened[output_data_labels[i]]['agent_entropies_flattened'] <= bad_entropy_cutoff
            flattened[output_data_labels[i]]['filter_wrong_and_confident'] = wrong & confident
        
        # add flattened room label list for each output dataset (e.g. "r", "-", "h", "...")
        if room_specific:
            for i in range(len(output_data_labels)):
                flattened[output_data_labels[i]]['room_labels_flattened'] = cls.get_roomlabels(output_data_list[i])

        # split data by timestep/location
        # split['test_unseen'][3] should give a 1-D numeric array of entropy values for time step 3
        # count['test_unseen'][3] should give a tuple (qualified counts, unqualified counts) for time step 3
        # categories is a set. {1, 2, ... max time step} or {"-", "h". "r", ...}
        ##splits = {}
        counts = {}
        categories = None
        for i in range(len(output_data_labels)):
            agent_entropies_flattened = flattened[output_data_labels[i]]['agent_entropies_flattened']
            if timestep_specific:
                categories_flattened = flattened[output_data_labels[i]]['timesteps_flattened']
            elif room_specific:
                categories_flattened = flattened[output_data_labels[i]]['room_labels_flattened']
            ##split_data, count_data = \
            _, count_data = \
                cls._filter_and_split_ent_by_decision_quality(
                    flattened_data=[agent_entropies_flattened, categories_flattened], 
                    filter=flattened[output_data_labels[i]]['filter_wrong_and_confident'])
            ##splits[output_data_labels[i]] = split_data
            counts[output_data_labels[i]] = count_data
            # we want to narrow down to the common set of categories between the different output datasets
            if categories is None:
                categories = set(categories_flattened)
            else:
                categories = categories & set(categories_flattened)

        # plot grouped bar chart below
        # order the categories first
        if timestep_specific:
            category_ids = sorted(list(categories))
            category_name = "timestep"
        elif room_specific:
            category_ids = sorted(list(categories))  # TODO deal too many room types
            category_name = "room tag"

        # confidently wrong in test seen, confidently wrong in test unseen
        arr_a = [[counts[output_data_labels[i]][cat][0] for cat in category_ids] \
            for i in range(len(output_data_labels))]
        arr_a = np.vstack((arr_a[0], arr_a[1]))
        # otherwise in test seen, otherwise in test unseen
        arr_b = np.array([[counts[output_data_labels[i]][cat][1] for cat in category_ids] \
            for i in range(len(output_data_labels))])
        arr_b = np.vstack((arr_b[0], arr_b[1]))
        assert len(arr_a) == len(output_data_labels)
        # plot normalized bar chart, plot raw count bar chart 
        for norm in (True, False):
            cls.plot_grouped_bar_comparison(
                category_ids=category_ids,
                arr_a=arr_a, 
                arr_b=arr_b,
                category_name=category_name,
                quality_labels = ["confidently wrong", "otherwise"],
                output_data_labels=output_data_labels,
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
            count_data {defaultdict} -- keys as category indices. values as counts of (qualified datapts, unqualified datapts)
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
                                        title="{}Entropy Distribution".format("Cross " if cross_ent_bool else ""),
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
                                                    title='Teacher {}({}), Agent {}({}), {}Entropy Distribution'.format(
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
            flattened[output_data_labels[i]]['teacher_targets_flattened'], \ 
            flattened[output_data_labels[i]]['agent_softmaxes_flattened'], \
            flattened[output_data_labels[i]]['timesteps_flattened'] = \
                cls.flatten_targets_softmaxes_timesteps(output_data_list[i], action_type, cross_ent_bool)

        # splits['test_seen'][forward idx=0] = {'agent_softmax_vals':[...], 'teacher_target_bools': [.....], 'time_steps':[...]}
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
        # if split by time step. should also split here, need to create `timed_binned`
        # binned['test_seen'][forward idx=0] = {'agent_softmax_avg':[..<num bin vals>..], 'agent_softmax_std':[..<num bin vals>..], 'teacher_target_avg':[..<num bin vals>..], 'teacher_target_std':[..<num bin vals>..]}
        binned = {}
        # TODO
        for i in range(len(output_data_labels)):
            # specific to both output data and action class
            for j in range(len(action_reference)):
                binned[output_data_labels[i]][j] = cls._bin_softmax_compute_stats(splits[output_data_labels[i]][j])

        # y axis = teacher avg and variance
        # y axis = agent avg and variance
        # x = bin processed



        pass

    @classmethod
    def plot_grouped_bar_comparison(cls, category_ids, arr_a, arr_b, category_name,
                                    quality_labels, output_data_labels,
                                    figsize=(20, 12), normalized=True,
                                    h_line=True, dataset_colors=None):
        """
        Compare a set test seen and test unseen outputs by quality of agent decisions.
        Using grouped bar chart.
        :param category_ids: numpy array or list shape (num_categories,).
                         e.g. [0, 1, 2, 3] for 4 timesteps
                         e.g. ['h', 'r', '-', ...] for room tags
        :param arr_a: numpy array (num output datasets, num categories)
        :param arr_b: numpy array (num output datasets, num categories)
        :param arr_b: string to label x-axis. Such as "time" or "room tag"
        :param quality_labels: 2-string tuple. e.g. ('confidently wrong', 'otherwise')
        :param output_data_labels: multi-string tuple. e.g. ('test_seen', 'test_unseen')
        :param figsize: numeric tuple.
        :param normalized: boolean. Normalize counts.
        :param h_line: boolean. Add hline for baseline experiment
        :param dataset_colors: array of strings. hexcode colors.
        :return:
        """
        assert (arr_a.shape == arr_b.shape and
                arr_a.shape[0] == len(output_data_labels) and
                arr_a.shape[1] == len(category_ids))
        if dataset_colors is not None:
            assert(len(dataset_colors) >= len(output_data_labels))

        rs = [np.arange(len(arr_a[0]))]
        bar_width = min(0.2, 0.5 / arr_a.shape[0])

        # b & w
        colors = ['#000000', '#ffffff']
        if dataset_colors is None:
            dataset_colors = REF_COLORS
        dataset_colors = np.array(dataset_colors)

        if normalized:
            tot = arr_a + arr_b
            arr_a, arr_b = arr_a / tot, arr_b / tot

        # figure boundary
        plt.subplots(figsize=figsize)
        if normalized:
            limit = 1.05
        else:
            limit = np.max(arr_a + arr_b) * 1.05
        plt.ylim(0, limit)

        p_exp = []
        p1, p2 = None, None
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

            hoffset = bar_width + bar_width / float(1.5)
            # h line
            if i == 0 and h_line:
                hspan = hoffset * len(output_data_labels)
                h = plt.hlines(y=arr_a[i], xmin=rs[i] - 0.2,
                               xmax=rs[i] + hspan,
                               colors='k', alpha=0.8,
                               linestyles='dotted', label='baseline')
            # horizontal shift to next trial
            rs.append(rs[i] + hoffset)

        # title
        plt.title("{} : {} \n {} - {} split".format(
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
        plt.legend((p1[0], p2[0]), quality_labels, fontsize=14,
                   loc='lower right')
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

