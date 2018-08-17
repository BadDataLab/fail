import os
from logger import Logger
from singleton_decorator import singleton
from defaults import default_postprocessor, default_preprocessor, default_preprocessor_inverse, default_target_picker

def testbed():
    return Testbed()


def init_testbed():
    tb = Testbed()
    tb.reset_testbed()
    return tb


@singleton
class Testbed:
    """
    Abstracts the running and evaluation of several attack instances.
    It also offers a convenient way of setting default behaviour across the performed experiments
    (e.g., when analyzing the same attack/dataset with different attacker knowledge/control constraints).
    """

    def __init__(self):
        self.logger = None
        self.preproc = None
        self.preproc_inverse = None
        self.postproc = None
        self.victim = None
        self.surrogate = None
        self.dataset = None
        self.craft_f = None
        self.target_picker = None
        self.apply_f = None
        self.experiments = []

    def __enter__(self):
        self.reset_testbed()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True

    def reset_testbed(self):
        self.logger = None
        self.preproc = None
        self.preproc_inverse = None
        self.postproc = None
        self.victim = None
        self.surrogate = None
        self.dataset = None
        self.craft_f = None
        self.target_picker = None
        self.apply_f = None
        self.experiments = []

    def register_dataset(self, dataset):
        self.dataset = dataset

    def register_surrogate(self, surrogate):
        self.surrogate = surrogate

    def register_victim(self, victim):
        self.victim = victim

    def register_logger(self, logger):
        self.logger = logger

    def new_experiment_log(self, labels, experiment_index):
        self.logger = Logger(os.path.join(self.out_dir, 'RESULTS_{}.txt'.format(experiment_index)), tags=labels)
        return self.logger

    def get_logger(self):
        if self.logger is None:
            self.logger = Logger(self.out_dir)

    def get_dataset(self):
        if self.dataset is None:
            print('No dataset set.')
        return self.dataset

    def get_preprocessor(self):
        if self.preproc is None:
            print('Using default preprocessor.')
            return default_preprocessor
        else:
            return self.preproc

    def get_preprocessor_inverse(self):
        if self.preproc_inverse is None:
            print('Using default preprocessor inverse.')
            return default_preprocessor_inverse
        else:
            return self.preproc_inverse

    def get_craft(self):
        if self.craft_f is None:
            print('No craft function set.')
        return self.craft_f

    def get_postprocessor(self):
        if self.postproc is None:
            print('Using default postporcessor.')
            return default_postprocessor
        else:
            return self.postproc

    def get_apply_function(self):
        if self.apply_f is None:
            print('No apply function set.')
        return self.apply_f

    def get_surrogate_model(self):
        if self.surrogate is None:
            print('No surrogate model set.')
        return self.surrogate

    def get_victim_model(self):
        if self.victim is None:
            print('No victim model set.')
        return self.victim

    def get_target_picker(self):
        if self.target_picker is None:
            print('Using default target picker (i.e., indiscriminate attack).')
            return default_target_picker
        else:
            return self.target_picker

    def register_experiment(self, e):
        self.experiments.append(e)

    def run_experiments(self, runs, out_dir):
        self.out_dir = out_dir
        for e in self.experiments:
            for rid in range(runs):
                e.run_once(rid)
