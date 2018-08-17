import numpy as np
import os
import sys
import copy

from testbed import testbed
from examples.stingray_experiments import StingRayUnconstrainedExperiment
from dataset import DrebinDataset

### DREBIN configs
DATA_DIR = '/Users/osuciu/Documents/datasets/drebin'
SAMPLING_SEED = 12552
TRAINING_SET_SIZE = -1
NO_OF_FEATURES = 500
DO_FEATURE_SELECTION = False
DO_SAMPLE = True
TARGET_LABEL = 1



def getSVM():
    from sklearn import svm
    C = 0.1
    RANDOM_STATE = 0
    clf = svm.LinearSVC(C=C,random_state=RANDOM_STATE)
    return clf

def getSGD():
    from sklearn import linear_model
    RANDOM_STATE = 0
    reg = linear_model.SGDClassifier(random_state=RANDOM_STATE)
    return reg

def getRF():
    from sklearn.ensemble import RandomForestClassifier 
    RANDOM_STATE = 0
    clf = RandomForestClassifier(random_state=RANDOM_STATE)
    return clf

def new_classifer(classifier,dataset):
    return SklearnClassifier(classifier,dataset) 


class SklearnClassifier:
    def __init__(self,clf):
        self.clf = clf

    def train(self,dataset):
        self.clf.fit(dataset.x, dataset.y)

    def test(self,dataset):
        test_predictions = self.clf.predict(dataset.x)
        return test_predictions

    def reset(self):
        newclf = copy.copy(self)
        return newclf

if __name__ == '__main__':
        tb = testbed()
    
        victim = SklearnClassifier(getSVM())
        surrogate = SklearnClassifier(getSVM())

        drebin = DrebinDataset(data_dir=DATA_DIR,training_size=TRAINING_SET_SIZE,reduce_features=DO_FEATURE_SELECTION, no_of_features=NO_OF_FEATURES, seed=SAMPLING_SEED,sample=DO_SAMPLE)
        
        tb.register_dataset(drebin)
        tb.register_victim(victim)
        tb.register_surrogate(surrogate)


        e1 = StingRayUnconstrainedExperiment(['some_label'])
        # e2 = JSMALimitedFeatureKnowledgeExperiment(border=4)
        tb.register_experiment(e1)

        tb.run_experiments(runs=2, out_dir=os.path.join('output', 'stingray-FAIL'))
        
