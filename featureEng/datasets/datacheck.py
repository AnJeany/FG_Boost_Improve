#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/06/28 17:51
@Author: Merc2
'''
import numpy as np
from pathlib import Path



class UCIDataset:
    def __init__(self, dataset, parent="C://Users/16499/Downloads/data_py/data//"):

        self.root = Path(parent) / dataset
        data_file = sorted(self.root.glob(f'{dataset}*.dat'))[0]
        label_file = sorted(self.root.glob('label*.dat'))[0]
        val_file = sorted(self.root.glob('validation*.dat'))[0]
        fold_index = sorted(self.root.glob('folds*.dat'))[0]
        self.dataX = np.loadtxt(data_file, delimiter=',')
        self.dataY = np.loadtxt(label_file, delimiter=',')
        self.validation = np.loadtxt(val_file, delimiter=',')
        self.folds_index = np.loadtxt(fold_index, delimiter=',')
        self.n_CV = self.folds_index.shape[1]
        types = np.unique(self.dataY)
        self.n_types = types.size
        # One hot coding for the target
        self.dataY_tmp = np.zeros((self.dataY.size, self.n_types))
        for i in range(self.n_types):
            for j in range(self.dataY.size):  # remove this loop
                if self.dataY[j] == types[i]:
                    self.dataY_tmp[j, i] = 1

    def getitem(self, CV):
        full_train_idx = np.where(self.folds_index[:, CV] == 0)[0]
        train_idx = np.where((self.folds_index[:, CV] == 0) & (self.validation[:, CV] == 0))[0]
        test_idx = np.where(self.folds_index[:, CV] == 1)[0]
        val_idx = np.where(self.validation[:, CV] == 1)[0]
        trainX = self.dataX[train_idx, :]
        trainY = self.dataY_tmp[train_idx, :]
        testX = self.dataX[test_idx, :]
        testY = self.dataY_tmp[test_idx, :]
        evalX = self.dataX[val_idx, :]
        evalY = self.dataY_tmp[val_idx, :]
        full_train_x = self.dataX[full_train_idx, :]
        full_train_y = self.dataY_tmp[full_train_idx, :]
        return trainX, trainY, evalX, evalY, testX, testY, full_train_x, full_train_y


# we add a empty '', to match the index of [1, len+1]
label_to_name =['', 'abalone', 'acute-inflammation', 'acute-nephritis', 'annealing', 'audiology-std', 'balance-scale', 'balloons', 'blood', 'breast-cancer', 'breast-cancer-wisc', 'breast-cancer-wisc-diag', 'breast-cancer-wisc-prog', 'breast-tissue', 'car', 'congressional-voting', 'conn-bench-sonar-mines-rocks', 'conn-bench-vowel-deterding', 'contrac', 'credit-approval', 'cylinder-bands', 'dermatology', 'echocardiogram', 'ecoli', 'energy-y1', 'energy-y2', 'fertility', 'flags', 'glass', 'haberman-survival', 'hayes-roth', 'heart-cleveland', 'heart-hungarian', 'heart-switzerland', 'heart-va', 'hepatitis', 'horse-colic', 'ilpd-indian-liver', 'ionosphere', 'iris', 'led-display', 'lenses', 'libras', 'lung-cancer', 'lymphography', 'mammographic', 'molec-biol-promoter', 'monks-1', 'monks-2', 'monks-3', 'oocytes_merluccius_nucleus_4d', 'oocytes_merluccius_states_2f', 'oocytes_trisopterus_nucleus_2f', 'oocytes_trisopterus_states_5b', 'parkinsons', 'pima', 'pittsburg-bridges-MATERIAL', 'pittsburg-bridges-REL-L', 'pittsburg-bridges-SPAN', 'pittsburg-bridges-T-OR-D', 'pittsburg-bridges-TYPE', 'planning', 'post-operative', 'primary-tumor', 'seeds', 'soybean', 'spect', 'spectf', 'statlog-australian-credit', 'statlog-german-credit', 'statlog-heart', 'statlog-image', 'statlog-vehicle', 'synthetic-control', 'teaching', 'tic-tac-toe', 'titanic', 'trains', 'vertebral-column-2clases', 'vertebral-column-3clases', 'wine', 'wine-quality-red', 'yeast', 'zoo', 'bank', 'blood']

def analysis_data():
    id_name=[]
    id_name.append(0)
    id_size =[]
    id_size.append(0)
    for gassid in range(1, 84):  #(params['data_num']+1)
        dataset = UCIDataset(label_to_name[gassid])
        train_x, train_y, eval_X, val_y, test_x, test_y, full_train_x, full_train_y = dataset.getitem(0)
        id_name.append(train_x.shape[1])
        id_size.append(train_x.shape[0])

    ind =[] # check which index of data that has over 10 features.
    for i, id in enumerate(label_to_name):
        # print(id, id_name[i])
        if id_name[i] > 10:
            print(id, ' ', id_size[i], '  ', i)
            ind.append(i)
    print(np.sum(np.array(id_name) > 10)) # --42
    print(ind)

analysis_data()
# usage

# datasets = UCIDataset('loc/to/datasets')
# trainX, trainY, evalX, evalY, testX, testY, full_train_x, full_train_y = datasets.getitem(0) # 0 is the CV index
# trainX is the training data
# trainY is the training label
# evalX is the validation data
# evalY is the validation label
# testX is the test data
# testY is the test label
# full_train_x is the full training data
# full_train_y is the full training label
