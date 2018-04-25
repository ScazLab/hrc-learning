from __future__ import unicode_literals
import numpy as np
import itertools as iter
import os
import errno
import re
from abc import ABCMeta, abstractmethod
from task_models.task import HierarchicalTask, \
    LeafCombination, SequentialCombination, \
    ParallelCombination, AlternativeCombination, \
    AbstractAction


def unique_rows(a):
    b = a.ravel().view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, unique_idx = np.unique(b, return_index=True)
    return a[unique_idx]


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


class HierarchicalTaskHMM(HierarchicalTask):
    """Tree representing a hierarchy of tasks which leaves are actions."""

    _metaclass_ = ABCMeta

    def __init__(self, root=None, name=None, num_feats_action=None,
                 feats='cumulative', supp_bhvs=None, obj_presence=None,
                 obj_count_idx=None, main_obj=None):
        super(HierarchicalTaskHMM, self).__init__(root)
        self.name = name
        self.num_obj = 0
        self.obj_names = dict()
        self.feat_names = dict()
        self.feats = feats
        self.num_feats_action = num_feats_action
        self.supp_bhvs = supp_bhvs
        self.obj_presence = obj_presence
        self.obj_count_idx = obj_count_idx
        self.main_obj = main_obj
        self.all_trajectories = []
        self.bin_trajectories = []
        self.bin_trajectories_test = []
        self.train_set_actions = []
        self.train_set_traj = []
        self.test_set_actions = []
        self.test_set_traj = []
        self.train_set_sb = []
        self.train_set_sb_actions = []
        self.train_set_sb_traj = []
        self.test_set_sb = []
        self.test_set_sb_traj = []
        self.test_set_sb_actions = []
        self.gen_dict()

    def gen_dict(self):
        regexp = r'( order)'
        self._gen_dict_rec(self.root, regexp)
        if self.feats == 'cumulative' or self.feats == 'shared':
            for feat_idx in range(self.num_feats_action):
                self.feat_names["feat_" + str(feat_idx)] = feat_idx
        else:
            self.feat_names = self.obj_names

    def _gen_dict_rec(self, node, regexp):
        if isinstance(node, LeafCombination):
            name = node.name
            rem = re.search(regexp, node.name)
            if rem:
                name = node.name[:rem.start()]
            if not(name in self.obj_names.keys()):
                self.obj_names[name] = self.num_obj
                self.num_obj += 1
        else:
            for child in node.children:
                self._gen_dict_rec(child, regexp)

    def _get_cum_feats_keys(self, traj):
        regexp = r'( order)'
        keys = [self.obj_names.get(key.name) if re.search(regexp, key.name) is None
                else self.obj_names.get(key.name[:re.search(regexp, key.name).start()])
                for key in traj]
        return keys

    @staticmethod
    def base_name(name):
        regexp = r'( order)'
        rem = re.search(regexp, name)
        if rem:
            name = name[:rem.start()]
        return name

    def gen_all_trajectories(self):
        self.all_trajectories = \
            self._gen_trajectories_rec(self.root)

    def _gen_trajectories_rec(self, node):
        """Generates all possible trajectories from an HTM.
        :returns: list of tuples of the following form
        [(proba, [LC1, LC2, ...]), (), ...]
        Each tuple represents a trajectory of the HTM,
        with the first element equal to the probability of that trajectory
        and the second element equal to the list of leaf combinations.
        """
        if isinstance(node, LeafCombination):
            return [(1, [node])]
        elif isinstance(node, ParallelCombination):
            return self._gen_trajectories_rec(node.to_alternative())
        elif isinstance(node, AlternativeCombination):
            children_trajectories = [(p * node.proba[c_idx], seq)
                                     for c_idx, c in enumerate(node.children)
                                     for p, seq in self._gen_trajectories_rec(c)]
            return children_trajectories
        elif isinstance(node, SequentialCombination):
            children_trajectories = [
                self._gen_trajectories_rec(c)
                for c_idx, c in enumerate(node.children)]
            new_trajectories = []
            product_trajectories = list(
                iter.product(*children_trajectories))
            for product in product_trajectories:
                probas, seqs = zip(*product)
                new_trajectories.append((float(np.product(probas)),
                                         list(iter.chain.from_iterable(seqs))))
            return new_trajectories
        else:
            raise ValueError("Reached invalid type during recursion.")

    def gen_bin_feats_traj(self):
        """Generates binary features for all possible trajectories from an HTM
        and writes them to file.
        Should only be called after calling gen_all_trajectories().
        """
        if self.all_trajectories:
            trajectories = list(zip(*self.all_trajectories))[1]
            bin_feats_init = np.array([0] * self.num_feats_action)
            for traj_idx, traj in enumerate(trajectories):
                bin_feats = np.tile(bin_feats_init,
                                    (len(set(traj)) + 1, 1))
                for node_idx, node in enumerate(traj):
                    self._gen_bin_feats_traj(node_idx, node, bin_feats,
                                             num_objs, num_types_leaves,
                                             traj=traj, user_prefs=user_prefs)
                self.bin_trajectories.append(bin_feats)
            self.bin_trajectories = np.array(self.bin_trajectories)
        else:
            raise ValueError("Cannot generate bin feats before generating all trajectories.")


    def _gen_bin_feats_traj(self, node_idx, node, traj_bin_feats,
                            num_objs, num_types_leaves,
                            traj=None, user_prefs=None):
        node_bin_feats_sb = self._gen_bin_feats_single(
            node.action.num_feats, node.action.feat_probs)
        action = self.obj_names.get(traj[node_idx].name)
        if action is None:
            action = self.obj_names.get(self.base_name(traj[node_idx].name))
        start_idx = action * self.num_feats_action
        end_idx = start_idx + self.num_feats_action
        traj_bin_feats[node_idx + 1, start_idx:end_idx] = node_bin_feats_sb

    @staticmethod
    def _gen_bin_feats_single(num_feats, feat_probs):
        """Generates binary features only for a leaf from the HTM.
        :param num_feats: number of features this object has (for now, same for all objects)
        :param feat_probs: list of probabilities of length = num_feats, where each prob
        is used to generate each of the binary features for this object (for now, each prob
        is generated using a uniform distribution)
        :returns: list of num_feats binary features generated based on the feat_probs list
        """
        if num_feats != len(feat_probs):
            raise ValueError("num_feats != len(feat_probs)"
                             "You should pass in prob values for all features.")
        bin_feats = np.random.binomial(1, feat_probs, size=num_feats)
        return bin_feats

