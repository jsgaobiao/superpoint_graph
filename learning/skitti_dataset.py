#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:45:16 2018
@author: landrieuloic
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import os
import functools
import torch
import torchnet as tnt
import h5py
import spg

def get_datasets(args, test_seed_offset=0):
    """ Gets training and test datasets. """

    # Load superpoints graphs
    testlist, trainlist, validlist = [], [], []
    # valid_names = ['000000.h5','000100.h5', '000200.h5','000300.h5','000400.h5','000500.h5', \
    #                '000600.h5','000700.h5','000800.h5','000900.h5','001000.h5','001100.h5','001200.h5','001300.h5','001400.h5', \
    #                '001500.h5','001600.h5','001700.h5','001800.h5','001900.h5','002000.h5','002100.h5','002200.h5','002300.h5']
    # data_set_list = [0,1,2,3,4,5,6,7,8,9,10,90,91,92,93]
    # for n in data_set_list:
    #     if n != args.cvfold:
    #         path = '{}/superpoint_graphs/{:0>2d}/'.format(args.SKITTI_PATH, n)
    #         for fname in sorted(os.listdir(path)):
    #             if fname.endswith(".h5") and not (args.use_val_set and fname in valid_names):
    #                 #training set
    #                 trainlist.append(spg.spg_reader(args, path + fname, True))
    #             if fname.endswith(".h5") and (args.use_val_set  and fname in valid_names):
    #                 #validation set
    #                 validlist.append(spg.spg_reader(args, path + fname, True))

    # train
    train_set_list = [90]
    for n in train_set_list:
        path = '{}/superpoint_graphs/{:0>2d}/'.format(args.SKITTI_PATH, n)
        #train set
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".h5"):
                trainlist.append(spg.spg_reader(args, path + fname, True))
    # val
    val_set_list = [8]
    for n in val_set_list:
        path = '{}/superpoint_graphs/{:0>2d}/'.format(args.SKITTI_PATH, n)
        #val set
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".h5"):
                validlist.append(spg.spg_reader(args, path + fname, True))
    # test
    test_set_list = [8, 9, 10]
    for n in test_set_list:
        path = '{}/superpoint_graphs/{:0>2d}/'.format(args.SKITTI_PATH, n)
        #test set
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".h5"):
                testlist.append(spg.spg_reader(args, path + fname, True))

    # Normalize edge features
    if args.spg_attribs01:
        trainlist, testlist, validlist, scaler = spg.scaler01(trainlist, testlist, validlist=validlist)
        
    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, db_path=args.SKITTI_PATH)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.SKITTI_PATH, test_seed_offset=test_seed_offset)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in validlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.SKITTI_PATH, test_seed_offset=test_seed_offset)), \
            scaler


def get_info(args):
    edge_feats = 0
    for attrib in args.edge_attribs.split(','):
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1
    if args.loss_weights == 'none':
        weights = np.ones((19,),dtype='f4')
    else:
        weights = h5py.File(args.SKITTI_PATH + "/parsed/class_count.h5")["class_count"][:].astype('f4')
        weights = weights[:,[i for i in range(6) if i != args.cvfold-1]].sum(1)
        weights = (weights+1).mean()/(weights+1)
    if args.loss_weights == 'sqrt':
        weights = np.sqrt(weights)
    weights = torch.from_numpy(weights).cuda() if args.cuda else torch.from_numpy(weights)
    return {
        'node_feats': 9 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'classes': 19,
        'class_weights': weights,
        'inv_class_map': {0:'unlabeled', 1:'car', 2:'bicycle', 3:'motorcycle', 4:'truck', 5:'other-vehicle', 6:'person', 7:'bicyclist', 8:'motorcyclist', 9:'road', 10:'parking', 11:'sidewalk', 12:'other-ground', \
            13:'building', 14:'fence', 15:'vegetation', 16:'trunk', 17:'terrain', 18:'pole', 19:'traffic-sign'},
    }

def preprocess_pointclouds(SKITTI_PATH):
    """ Preprocesses data by splitting them by components and normalizing."""
    class_count = np.zeros((19,15),dtype='int')
    data_set_list = [0,1,2,3,4,5,6,7,8,9,10,90,91,92,93]
    class_n = 0
    for n in data_set_list:
        pathP = '{}/parsed/{:0>2d}/'.format(SKITTI_PATH, n)
        pathD = '{}/features_supervision/{:0>2d}/'.format(SKITTI_PATH, n)
        pathC = '{}/superpoint_graphs/{:0>2d}/'.format(SKITTI_PATH, n)
        if not os.path.exists(pathP):
            os.makedirs(pathP)
        random.seed(n)

        fList = os.listdir(pathC)
        fList.sort()
        for file in fList:
            print("dataset {} : {} ".format(n, file))
            if file.endswith(".h5"):
                f = h5py.File(pathD + file, 'r')
                xyz = f['xyz'][:]
                rgb = f['rgb'][:].astype(np.float)
                
                labels = f['labels'][:]
                hard_labels = np.argmax(labels[:,1:],1)
                label_count = np.bincount(hard_labels, minlength=19)
                class_count[:,class_n] = class_count[:,class_n] + label_count
                
                e = (f['xyz'][:,2][:] -  np.min(f['xyz'][:,2]))/ (np.max(f['xyz'][:,2]) -  np.min(f['xyz'][:,2]))-0.5

                rgb = rgb/255.0 - 0.5
                
                xyzn = (xyz - np.array([30,0,0])) / np.array([30,5,3])
                
                lpsv = np.zeros((e.shape[0],4))

                P = np.concatenate([xyz, rgb, e[:,np.newaxis], lpsv, xyzn], axis=1)

                f = h5py.File(pathC + file, 'r')
                numc = len(f['components'].keys())

                with h5py.File(pathP + file, 'w') as hf:
                    hf.create_dataset(name='centroid',data=xyz.mean(0))
                    for c in range(numc):
                        idx = f['components/{:d}'.format(c)][:].flatten()
                        if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                            ii = random.sample(range(idx.size), k=10000)
                            idx = idx[ii]

                        hf.create_dataset(name='{:d}'.format(c), data=P[idx,...])
        class_n += 1
    path = '{}/parsed/'.format(SKITTI_PATH)
    data_file = h5py.File(path+'class_count.h5', 'w')
    data_file.create_dataset('class_count', data=class_count, dtype='int')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    parser.add_argument('--SKITTI_PATH', default='datasets/sequences')
    args = parser.parse_args()
    preprocess_pointclouds(args.SKITTI_PATH)