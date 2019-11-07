import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

COLORS = ["c", "m", "y", "g", "b"]


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
        # compute cross entropy instead of entropy
        if cross_ent_bool:
            agent_ent_str.replace("ent", "cross_ent")
        return teacher_target_str, agent_argmax_str, agent_ent_str

    @classmethod
    def flatten_targets_argmaxes_entropies(cls, data, action_type, cross_ent_bool=False):
        """
        flatten targets, argmaxes and entropy values.
        :param data: an OutputData data object.
        :param action_type: string. should contain pattern to specific whether nav or ask. initial or final (nav)
        :param cross_ent_bool: (o) boolean. True if we want to look at cross-entropy instead of entropy
        :return: 3 numpy arrays. each of shape (sum_datapt(timesteps in datapt i), )
        """
        teacher_target_str, agent_argmax_str, agent_ent_str = cls._parse_action_type(action_type, cross_ent_bool)
        teacher_targets_flattened = np.array([target for datapt in data for target in datapt[teacher_target_str]])
        agent_argmaxes_flattened = np.array([argmax for datapt in data for argmax in datapt[agent_argmax_str]])
        agent_entropies_flattened = np.array([entropy for datapt in data for entropy in datapt[agent_ent_str]])
        return teacher_targets_flattened, agent_argmaxes_flattened, agent_entropies_flattened

    @classmethod
    def _split_ent_data(cls, flattened_data):
        """
        split data by tuple keys (teacher gold target, agent predicted argmax target)
        :param flattened_data: a list of three array elements returned from  cls.flatten_targets_argmaxes_entropies()
        :param action_type: string. should contain pattern to specific whether nav or ask. initial or final (nav)
        :param cross_ent_bool: (o) boolean. True if we want to look at cross-entropy instead of entropy
        :return split_data : a default dictionary object with tuple keys (teacher gold target,
                            agent predicted argmax target)
        """
        teacher_targets_flattened, agent_argmaxes_flattened, agent_entropies_flattened = flattened_data[0], \
                                                                                         flattened_data[1], \
                                                                                         flattened_data[2]
        split_data = defaultdict(list)
        for i in range(len(teacher_targets_flattened)):
            split_data[(teacher_targets_flattened[i], agent_argmaxes_flattened[i])].append(agent_entropies_flattened[i])
        return split_data

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
            plt.hist(arr, color=COLORS[i], label=label_list[i], **kwargs)
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend()
        plt.show()

    @classmethod
    def plot_by_action_type(cls, output_data_list, output_data_labels, action_type, cross_ent_bool=False,
                            action_reference=None):
        """
        Plot and visualize all histograms related to an action type to observe agent entropy distribution.
        One histogram is plot for very (teacher gold target, agent predicted argmax) combination.
        :param output_data_list: a list of 1+ OutputData objects. Usually one for test seen and one for test unseen.
        :param action_type: string. should contain pattern to specific whether nav or ask. initial or final (nav)
        :param cross_ent_bool: (o) boolean. True if we want to look at cross-entropy instead of entropy
        :param action_reference: (o). A list/array of strings each describing an action by the index.
                                 e.g. ['left', 'right', 'up', 'down', 'forward'] for nav action type
                                 e.g. ["dont_ask", "ask"] for ask action type
        """
        print("Action Type : {}".format(action_type))
        if not action_reference:
            if "nav" in action_type:
                action_reference = ['left', 'right', 'up', 
                'down', 'forward', '<end>']
            elif "ask" in action_type:
                action_reference = ["dont_ask", "ask"]
        print("Action Reference : {}".format(action_reference))

        # flattened data into arrays of shape (sum_datapt(num time steps at datapt i), )
        # flattened['test_unseen']['teacher_targets_flattened'] is an array
        assert len(output_data_list) == len(output_data_labels)
        flattened = defaultdict(lambda: defaultdict(list))
        for i in range(len(output_data_labels)):
            flattened[output_data_labels[i]]['teacher_targets_flattened'], \
            flattened[output_data_labels[i]]['agent_argmaxes_flattened'], \
            flattened[output_data_labels[i]]['agent_entropies_flattened'] = \
                cls.flatten_targets_argmaxes_entropies(output_data_list[i], action_type, cross_ent_bool)

        # plot overall entropy histogram (all action targets)
        kwargs = dict(alpha=0.5, bins=30, density=True, stacked=True)
        cls.plot_overlapping_histograms(arr_list=[flattened[label]['agent_entropies_flattened'] for label in flattened],
                                        label_list=output_data_labels, # e.g. ['test_seen', 'test_unseen'],
                                        title="{}Entropy Distribution".format("Cross " if cross_ent_bool else ""),
                                        xlab="Per agent decision entropy",
                                        ylab="Density",
                                        **kwargs)

        # split data by key (teacher gold target, agent predicted argmax target) using self._split_ent_data()
        # splits['unseen'][(teacher target=1, agent argmax=1)] should give an array
        splits = {}
        for i in range(len(output_data_labels)):
            splits[output_data_labels[i]] = cls._split_ent_data(flattened_data=
                [flattened[output_data_labels[i]]['teacher_targets_flattened'],
                 flattened[output_data_labels[i]]['agent_argmaxes_flattened'],
                 flattened[output_data_labels[i]]['agent_entropies_flattened']
                 ])

        actions = range(len(action_reference))
        for teacher_tar_idx in actions:
            for agent_argmax_idx in actions:
                cls.plot_overlapping_histograms(arr_list=[splits[label][(teacher_tar_idx, agent_argmax_idx)] for label in splits],
                                                label_list=output_data_labels,  # e.g. ['test_seen', 'test_unseen'],
                                                title='Teacher {}({}), Agent {}({}), {}Entropy Distribution'.format(
                                                    teacher_tar_idx, action_reference[teacher_tar_idx],
                                                    agent_argmax_idx, action_reference[agent_argmax_idx],
                                                    "Cross " if cross_ent_bool else ""),
                                                xlab="Per agent decision entropy",
                                                ylab="Density",
                                                **kwargs)


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