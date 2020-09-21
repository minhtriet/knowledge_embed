import os
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import itertools
import random

class RelationDataset(Dataset):
    def __init__(self, data_directory, typed=False, no_negsamples=10):
        self.NO_NEGSAMPLES = no_negsamples
        random.seed(42)
        self.data_directory = data_directory
        entityFile = 'entity2id.txt';
        relationFile = 'relation2id.txt';
        typeFile = 'type2id.txt'
        if all('typed_' + dtype + '2id.txt' in os.listdir(self.data_directory) for dtype in ['train', 'valid', 'test']):
            trainFile = 'typed_train2id.txt';
            devFile = 'typed_valid2id.txt';
            testFile = 'typed_test2id.txt'
        else:
            trainFile = 'train2id.txt';
            devFile = 'valid2id.txt';
            testFile = 'test2id.txt'
        typeConstraintsFile = 'type_constrain.txt'
        # build entity & relation index lookup
        # assumes file2idx.txt has numlines as the first line, and format name\tidx in rest
        with open(os.path.join(self.data_directory, entityFile), 'r') as f:
            entity2idx = {line.split()[0]: int(line.split('\t')[1]) for line in f.readlines()[1:]}
        with open(os.path.join(self.data_directory, relationFile), 'r') as f:
            relation2idx = {line.split()[0]: int(line.split('\t')[1]) for line in f.readlines()[1:]}
        if typeFile in os.listdir(self.data_directory):
            with open(os.path.join(self.data_directory, typeFile), 'r') as f:
                type2idx = {line.split()[0]: int(line.split('\t')[1]) for line in f.readlines()[1:]}
        with open(os.path.join(self.data_directory, trainFile), 'r') as f:
            trainData = [[int(i) for i in line.split()] for line in f.readlines()[1:]]
        with open(os.path.join(self.data_directory, devFile), 'r') as f:
            devData = [[int(i) for i in line.split()] for line in f.readlines()[1:]]
        with open(os.path.join(self.data_directory, testFile), 'r') as f:
            testData = [[int(i) for i in line.split()] for line in f.readlines()[1:]]
        if typeConstraintsFile in os.listdir(self.data_directory):
            with open(os.path.join(self.data_directory, typeConstraintsFile), 'r') as f:
                lines = f.readlines()
            typeConstraints_left = {line[0]: line[2:] for line in [[int(i) for i in \
                                                                    l.split()] for j, l in enumerate(lines[1:]) if
                                                                   j % 2 == 0]}
            typeConstraints_right = {line[0]: line[2:] for line in [[int(i) for i in \
                                                                     l.split()] for j, l in enumerate(lines[1:]) if
                                                                    j % 2 == 1]}
        else:
            typeConstraints_left = defaultdict(range(len(entity2idx)))
            typeConstraints_right = defaultdict(range(len(entity2idx)))
        # compile triplets filter -- a defaultValueDict that returns
        # True if the triplet occurs in the training, dev, or test data.
        tripletsFilter = defaultdict(default_factory=False)
        for line in trainData + devData + testData:
            tripletsFilter[(line[0], line[2], line[1])] = True
        e1s_train, rs_train, e2s_train = [], [], []
        if typed:
            e1types_train = []
        for line in trainData:
            e1s_train.append(line[0]);
            rs_train.append(line[2]);
            e2s_train.append(line[1])
            if typed: e1types_train.append(line[3])
        trainData = [e1s_train, rs_train, e2s_train]
        if typed: trainData.append(e1types_train)
        e1s_dev, rs_dev, e2s_dev = [], [], []
        if typed: e1types_dev = []
        for line in devData:
            e1s_dev.append(line[0])
            rs_dev.append(line[2])
            e2s_dev.append(line[1])
            if typed: e1types_dev.append(line[3])
        devData = [e1s_dev, rs_dev, e2s_dev]
        if typed: devData.append(e1types_dev)
        e1s_test, rs_test, e2s_test = [], [], []
        if typed: e1types_test = []
        for line in testData:
            e1s_test.append(line[0])
            rs_test.append(line[2])
            e2s_test.append(line[1])
            if typed: e1types_test.append(line[3])
        testData = [e1s_test, rs_test, e2s_test]
        if typed: testData.append(e1types_test)
        self.entity2idx = entity2idx
        self.relation2idx = relation2idx
        data = {'filter': tripletsFilter, \
                'candidates_l': typeConstraints_left, 'candidates_r': typeConstraints_right}
        if typed:
            self.data['type2idx'] = type2idx
        else:
            self.train_data, self.val_data, self.test_data = [D[:3] for D in [trainData, devData,
                                                                              testData]]  # remove type info if it is there
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        e1_neg = np.random.choice(len(self.entity2idx), self.NO_NEGSAMPLES)
        e2_neg = np.random.choice(len(self.entity2idx), self.NO_NEGSAMPLES)
        e1_pos, r, e2_pos = self.train_data[0][idx], self.train_data[1][idx], self.train_data[2][idx]
        return np.concatenate(([e1_pos, r, e2_pos], e1_neg, e2_neg))
