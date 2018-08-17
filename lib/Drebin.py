import os
import csv
import json
import random
from collections import defaultdict
import numpy as np
from sklearn import svm, cross_validation
from scipy.sparse import csc_matrix


def sampleBenign(apps, seed):
    appType = defaultdict(list)
    for h, isMal in apps:
        appType[isMal].append(h)

    malCount = len(appType.get(True))
    random.seed(seed)
    appType[False] = random.sample(appType[False], malCount)
    results = []
    for isMal, hashList in appType.items():
        for h in hashList:
            results.append([h, isMal])

    random.shuffle(results)
    return results


def readData(featSet = None, sample = False, seed=1234):
  # read malware hashes
  with open(os.path.join(DATA_DIR,"sha256_family.csv"), "r") as f:
    reader = csv.DictReader(f)
    malwareSet = [item.get("sha256") for item in reader]
  malwareSet = set(malwareSet)

  # read app hashes
  path = "feature_vectors/"
  
  appHashes = json.loads(open(os.path.join(DATA_DIR,'appHashes'),'r').read())
  appHashes = map(lambda x: [x, x in malwareSet], appHashes)

  if sample:
    appHashes = sampleBenign(appHashes,seed)

  labels = []
  count = 0
  rows = []
  cols = []
  index = 0
  featDict = {}

  print ('# apps:',len(appHashes))

  for h, isMal in appHashes:
    with open(os.path.join(DATA_DIR,path, h), "r") as f:
      feats = [feat for feat in f.read().split("\n") if feat]

    if featSet:
      feats = list(set(feats) & featSet)

    for f in feats:
      if f not in featDict:
        featDict[f] = index
        index += 1

      rows.append(count)
      cols.append(featDict.get(f))

    count += 1
    labels.append(int(isMal))

  data = [1 for i in range(len(rows))]
  features = csc_matrix((data, (rows, cols)), shape = (count, index))

  labels = np.asarray(labels)
  appHashes = np.asarray(appHashes)

  return features, labels, featDict, appHashes


def train():
  features, labels, featDict, appHashes = readData(sample = True)
  print ("done reading")
  clf = svm.SVC(kernel='linear', C=1)
  scores = cross_validation.cross_val_score(clf, features, labels, cv=5)
  print (scores)

if __name__ == 'main':
    train()



