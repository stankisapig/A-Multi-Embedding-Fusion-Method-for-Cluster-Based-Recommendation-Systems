# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 07:27:32 2020

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

#for user in range(len(train_data)):
#    current_user = user
#    for item in range(len(train_data[user])):
#        current_item = train_data[user][item]
#        ui_vector = lgbm_index_embedding(current_user, current_item)
#        current_gt = train_gt[current_user][item]


#train_x = []
#train_y = []
#for user in tqdm(range(len(train_data))):
#    current_user = user
#    for item in range(len(train_data[user])):
#        current_item = train_data[user][item]
#        ui_vector = lgbm_index_embedding(current_user, current_item)
#        current_gt = train_gt[current_user][item]
#        train_x.append(ui_vector)
#        train_y.append(current_gt)
#DATASET_DIR = 'pcity_data/'
train_x = np.load('pcity_data/Local_x_YELPP.npy')
train_y = np.load('pcity_data/Local_y_YELPP.npy')

#np.save(os.path.join(DATASET_DIR, 'Local_x_YELPP.npy'), np.array(train_x))
#np.save(os.path.join(DATASET_DIR, 'Local_y_YELPP.npy'), np.array(train_y))

model = lgb.LGBMClassifier(
        boosting_type = 'gbdt',
        n_estimators = 200,
        learning_rate =0.1,
        num_leaves = 31, 
        max_depth = -1,
        subsample = 0.9, #bagging_fraction
        colsample_bytree = 0.8,
        )
model.fit(train_x, train_y)
#model.booster_.save_model('Concate_Model_YELPM.h5')

#params = {    
#          'boosting_type': 'gbdt',
#          'n_estimators': 200,
#          'learning_rate':0.01,
#          'num_leaves':31, 
#          'max_depth': -1,
#          'subsample': 0.9, #bagging_fraction
#          'colsample_bytree': 0.8, 
#    }

top_k = [5,10,15,20,25,30]
recall = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
precision = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
num_gt = 5
final_test_x = []
final_test_y = []
all_dist_table = []
all_pred = []
for user in tqdm(range(len(user_test))):
    test_x = []
    test_y = []
    current_user = user
    for item in range(len(user_test[user])):
        current_item = user_test[user][item]
        ui_vector = lgbm_index_embedding(current_user, current_item)
        current_gt = train_gt[current_user][item]
        test_x.append(ui_vector)
        test_y.append(current_gt)
    pred = model.predict(test_x, test_y)
    dist_table = np.argsort(pred)
    all_dist_table.append(dist_table)
    all_pred.append(pred)
    for idx,k in enumerate(top_k):
        top = top_k[idx]
        current_recall, current_precision = lgbm_eva(dist_table, top, num_gt)
        recall[idx]+=current_recall
        precision[idx]+=current_precision
    final_test_x.append(test_x)
    final_test_y.append(test_y)
#np.save(os.path.join(opts.DATASET_DIR, opts.ALL_TEST,'LCtest_x_YELPP.npy'), np.array(final_test_x))
#np.save(os.path.join(opts.DATASET_DIR, opts.ALL_TEST,'LCtest_y_YELPP.npy'), np.array(final_test_y))
#np.save(os.path.join(opts.DATASET_DIR, 'distance/GC_all_pred_YELPP.npy'), np.array(all_pred))
#np.save(os.path.join(opts.DATASET_DIR, 'distance/GC_dist_table_YELPP.npy'), np.array(all_dist_table))
for idx,i in enumerate(top_k):
    print('topN:',i)
    print('recall:',recall[idx]/466)
    print('precision:',precision[idx]/466)





