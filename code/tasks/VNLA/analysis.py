import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats


class OutputData(object):

    def __init__(self, output_data):
        """output_data : json object loaded from output/.json file"""
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
        nav_ent_initial_list = [cls.compute_entropy(time_step) for time_step in datapt['agent_nav_logits_initial']]
        nav_ent_final_list = [cls.compute_entropy(time_step) for time_step in datapt['agent_nav_logits_final']]
        ask_ent_final_list = [cls.compute_entropy(time_step) for time_step in datapt['agent_ask_logits']]
        return nav_ent_initial_list, nav_ent_final_list, ask_ent_final_list

    @classmethod
    def compute_cross_entropy_per_datapt(cls, datapt):
        num_decisions = len(datapt['trajectory']) - 1
        nav_cross_ent_initial_list = [
            cls.compute_cross_entropy(datapt['agent_nav_logits_initial'][i], datapt['teacher_nav'][i]) for i in
            range(num_decisions)]
        nav_cross_ent_final_list = [
            cls.compute_cross_entropy(datapt['agent_nav_logits_final'][i], datapt['teacher_nav'][i]) for i in
            range(num_decisions)]
        ask_cross_ent_final_list = [cls.compute_cross_entropy(datapt['agent_ask_logits'][i], datapt['teacher_ask'][i])
                                    for i in range(num_decisions)]
        return nav_cross_ent_initial_list, nav_cross_ent_final_list, ask_cross_ent_final_list

    def compute_entropy_output_data(self):
        for i in range(len(self.output_data)):
            nav_ent_initial_list, nav_ent_final_list, ask_ent_final_list = \
                self.compute_entropy_per_datapt(self.output_data[i])
            self.output_data[i]['agent_nav_ent_initial'] = nav_ent_initial_list
            self.output_data[i]['agent_nav_ent_final'] = nav_ent_final_list
            self.output_data[i]['agent_ask_ent'] = ask_ent_final_list

    def compute_cross_entropy_output_data(self):
        for i in range(len(self.output_data)):
            nav_cross_ent_initial_list, nav_cross_ent_final_list, ask_cross_ent_final_list = \
                self.compute_cross_entropy_per_datapt(self.output_data[i])
            self.output_data[i]['agent_nav_cross_ent_initial'] = nav_cross_ent_initial_list
            self.output_data[i]['agent_nav_cross_ent_final'] = nav_cross_ent_final_list
            self.output_data[i]['agent_ask_cross_ent'] = ask_cross_ent_final_list

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


class PlotData(object):

    def __init__(self, output_data):
        self.output_data = output_data

    def split_ent_data(self, action_type, cross_ent=False):

        if 'nav' in action_type:
            teacher_target_str = 'teacher_nav'
            agent_argmax_str = 'agent_nav_argmax'
            agent_ents_str = 'agent_nav_ent'
            if 'initial' in action_type:
                agent_argmax_str += "_initial"
                agent_ents_str += "_initial"
            elif 'final' in action_type:
                agent_argmax_str += "_final"
                agent_ents_str += "_final"
        elif 'ask' in action_type:
            teacher_target_str = 'teacher_ask'
            agent_argmax_str = 'agent_ask'
            agent_ents_str = "agent_ask_ent"

        if cross_ent:
            agent_ents_str.replace("ent", "cross_ent")

        teacher_targets = np.array([target for datapt in self.output_data for target in datapt[teacher_target_str]])
        agent_argmaxes = np.array([argmax for datapt in self.output_data for argmax in datapt[agent_argmax_str]])
        agent_ents = np.array([ent for datapt in self.output_data for ent in datapt[agent_ents_str]])

        categories = set(teacher_targets)
        ent_by_teacher_tar_agent_argmax = defaultdict(list)

        for i in range(len(teacher_targets)):
            ent_by_teacher_tar_agent_argmax[(teacher_targets[i], agent_argmaxes[i])].append(agent_ents[i])

        return ent_by_teacher_tar_agent_argmax

    @classmethod
    def plot_agent_entropy(cls, teacher_target, agent_argmax, ent_split_data, cross_ent=False):
        """
        teacher_target : integer
        agent_armax : integer
        """
        data_to_plot = ent_split_data[(teacher_target, agent_argmax)]
        print("count = {}".format(len(data_to_plot)))
        plt.figure(figsize=(12, 6))
        plt.hist(data_to_plot, bins=30)
        plt.title('Teacher {}, Agent {}, {} Entropy Distribution'.format(teacher_target, agent_argmax,
                                                                         "Cross" if cross_ent else ""))
        plt.xlabel('Agent Decision Entropy')
        plt.show()

    def plot_by_action_type(self, action_type, cross_ent=False):

        ent_split_data = self.split_ent_data(action_type, cross_ent)

        sorted_key_pairs = sorted(list(ent_split_data), key=lambda x: (x[0], x[1]))
        target_set = set([key for keypair in sorted_key_pairs for key in keypair])

        for teacher_target in target_set:

            for agent_argmax in target_set:
                self.plot_agent_entropy(teacher_target, agent_argmax, ent_split_data, cross_ent)