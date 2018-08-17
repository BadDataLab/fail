from testbed import testbed
from lib import Drebin
#
import sys
import os
import json
import copy
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit


class Data:
    def __init__(self, xs, ys):
        self.x = xs
        self.y = ys

    def __len__(self):
        return len(self.x)

    def getx(self,idx):
        return self.x[idx:idx+1]

    def get(self,idxs):
        return Data(self.x[idxs],self.y[idxs])


class Dataset:

    def __init__(self, xs, ys, train_pct=0.7, test_pct=0.2):
        train_ix = int(train_pct * len(xs))
        test_ix = train_ix + int(test_pct * len(xs))

        self.data_train = Data(xs[:train_ix], ys[:train_ix])
        self.data_test = Data(xs[train_ix:test_ix], ys[train_ix:test_ix])
        self.data_craft = Data(xs[test_ix:], ys[test_ix:])

    def train(self):
        return self.data_train

    def test(self):
        return self.data_test

    def craft(self):
        return self.data_craft




class DrebinDataset:
    def __init__(self, data_dir, training_size=-1, reduce_features=False, no_of_features=500, seed=1234, sample=True):
        Drebin.DATA_DIR = data_dir

        features, labels, featDict, appHashes = Drebin.readData(sample=sample, seed=seed)

        print ('Done read')
    
        total_no_samples = labels.shape[0]

        if training_size is -1:
            training_size = (int) (0.66 * total_no_samples)

        sss = StratifiedShuffleSplit(labels, 1, train_size=training_size,test_size=total_no_samples-training_size, random_state=seed)
        train, test = list(sss)[0]

        train_features, test_features = features[train], features[test]
        train_labels, test_labels = labels[train], labels[test]
        train_hashes, test_hashes = appHashes[train], appHashes[test]

        print ('Done split')

        if reduce_features:
            train_features,test_features,selected_features_indices = _feature_selection(train_features, train_labels, test_features, no_of_features)
            selected_features = filter(lambda e: e[1] in selected_features_indices, featDict.items())
            selected_features = sorted(selected_features,key=lambda e: e[1])
            selected_features = map(lambda e: e[0],selected_features)
        else:
            selected_features = map(lambda e: e[0],sorted(featDict.items(),key=lambda e: e[1]))


        print ('Done FS')

        train_features = train_features
        train_labels = train_labels
        test_features = test_features
        test_labels = test_labels


        self.selected_features = selected_features
        self.train_hashes = train_hashes
        self.test_hashes = test_hashes
    
        self.data_train = Data(train_features, train_labels)
        self.data_test = Data(test_features, test_labels)
        self.data_craft = Data(test_features, test_labels)


    def train(self):
        return self.data_train

    def test(self):
        return self.data_test

    def craft(self):
        return self.data_craft


