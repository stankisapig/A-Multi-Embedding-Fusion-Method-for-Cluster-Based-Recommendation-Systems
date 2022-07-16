# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 20:00:29 2020

@author: stanley
"""

import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from data import load_data, get_lgbm_train, lgbm_index_embedding, lgbm_eva
from tqdm import tqdm
import os

from args import parser
opts = parser()

usernum, itemnum, train_pos, neg, get_gt, user_test = load_data()

train_data, train_gt = get_lgbm_train(train_pos, neg)

LC_all_pred = np.load(os.path.join(opts.DATASET_DIR, 'distance/LC_all_pred_YELPP.npy'),allow_pickle=True)
LC_dist_table = np.load(os.path.join(opts.DATASET_DIR, 'distance/LC_dist_table_YELPP.npy'),allow_pickle=True)

GC_all_pred = np.load(os.path.join(opts.DATASET_DIR, 'distance/GC_all_pred_YELPP.npy'),allow_pickle=True)
GC_dist_table = np.load(os.path.join(opts.DATASET_DIR, 'distance/GC_dist_table_YELPP.npy'),allow_pickle=True)

LE_all_pred = np.load(os.path.join(opts.DATASET_DIR, 'distance/LE_all_pred_YELPP.npy'),allow_pickle=True)
LE_dist_table = np.load(os.path.join(opts.DATASET_DIR, 'distance/LE_dist_table_YELPP.npy'),allow_pickle=True)

GE_all_pred = np.load(os.path.join(opts.DATASET_DIR, 'distance/GE_all_pred_YELPP.npy'),allow_pickle=True)
GE_dist_table = np.load(os.path.join(opts.DATASET_DIR, 'distance/GE_dist_table_YELPP.npy'),allow_pickle=True)

Final_all_pred = []
Final_all_pred_norm = []
for LC,GC,LE,GE in zip(LC_all_pred,GC_all_pred,LE_all_pred,GE_all_pred):
    Local = np.add(LC, LE)
    Global = np.add(GC, GE)
    Final = np.add(Local, Global)
    Final_all_pred.append(Final)
    min_val = min(Final)
    max_val = max(Final)
    len_range = max(Final) - min(Final)
    Final_norm = []
    for val in Final:
        norm_val= (val-min_val)/len_range
        Final_norm.append(norm_val)
    Final_norm = np.array(Final_norm)
    Final_all_pred_norm.append(Final_norm)
    
Final_dist_table = []
for user in Final_all_pred:
    dist_table = np.argsort(user)
    Final_dist_table.append(dist_table)

Final_dist_table_norm = []
for user in Final_all_pred_norm:
    dist_table = np.argsort(user)
    Final_dist_table_norm.append(dist_table)

top_k = [5,10,15,20,25,30]
recall = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
precision = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
num_gt = 5

for user in tqdm((Final_dist_table_norm)):
    for idx,k in enumerate(top_k):
        top = top_k[idx]
        current_recall, current_precision = lgbm_eva(user, top, num_gt)
        recall[idx]+=current_recall
        precision[idx]+=current_precision
        
for idx,i in enumerate(top_k):
    print('topN:',i)
    print('recall:',recall[idx]/466)
    print('precision:',precision[idx]/466)

