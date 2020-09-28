import os
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import itertools
import random

class RelationDataset(Dataset):
    def __init__(self, data_directory, slice, typed=False, no_negsamples=10):
        self.NO_NEGSAMPLES = no_negsamples
        self.slice = slice
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
        # build entity & relation index lookup
        # assumes file2idx.txt has numlines as the first line, and format name\tidx in rest
        with open(os.path.join(self.data_directory, entityFile), 'r') as f:
            entity2idx = {line.split()[0]: int(line.split('\t')[1]) for line in f.readlines()[1:]}
        with open(os.path.join(self.data_directory, relationFile), 'r') as f:
            relation2idx = {line.split()[0]: int(line.split('\t')[1]) for line in f.readlines()[1:]}
        if typeFile in os.listdir(self.data_directory):
            with open(os.path.join(self.data_directory, typeFile), 'r') as f:
                type2idx = {line.split()[0]: int(line.split('\t')[1]) for line in f.readlines()[1:]}
        if slice=="train":
            with open(os.path.join(self.data_directory, trainFile), 'r') as f:
                raw_data = [[int(i) for i in line.split()] for line in f.readlines()[1:]]
        elif slice=="dev":
            with open(os.path.join(self.data_directory, devFile), 'r') as f:
                raw_data = [[int(i) for i in line.split()] for line in f.readlines()[1:]]
        elif slice=="test":
            with open(os.path.join(self.data_directory, testFile), 'r') as f:
                raw_data = [[int(i) for i in line.split()] for line in f.readlines()[1:]]
        else: raise ValueError("Wrong slice")
        # compile triplets filter -- a defaultValueDict that returns
        # True if the triplet occurs in the training, dev, or test data.
        e1s_train, rs_train, e2s_train = [], [], []
        e1types_train = []
        for line in raw_data:
            e1s_train.append(line[0]);
            rs_train.append(line[2]);
            e2s_train.append(line[1])
            if typed:
                e1types_train.append(line[3])
        raw_data = [e1s_train, rs_train, e2s_train]
        if typed: raw_data.append(e1types_train)
        self.entity2idx = entity2idx
        self.relation2idx = relation2idx
        if typed:
            self.types = type2idx
        else:
            self.data = raw_data[:3]

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        if self.NO_NEGSAMPLES > 0:
            e1_neg = random.sample(list(self.entity2idx.values()), self.NO_NEGSAMPLES)
            e2_neg = random.sample(list(self.entity2idx.values()), self.NO_NEGSAMPLES)
            e1_pos, r, e2_pos = self.data[0][idx], self.data[1][idx], self.data[2][idx]
            return np.array(list(zip([e1_pos] + e1_neg, itertools.repeat(r), [e2_pos] + e2_neg)))
        return np.array([[self.data[0][idx], self.data[1][idx], self.data[2][idx]]])
