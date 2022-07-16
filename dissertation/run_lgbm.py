# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:21:15 2020

@author: stanley
"""


import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from data import load_data, get_lgbm_train_input

from args import parser
opts = parser()


usernum, itemnum, train_positive, negative, get_gt, user_test = load_data()
#usernum, itemnum, train_pos, neg, get_gt, user_test = load_data()
#u,i,j = get_embedding_input(train_positive, negative)
u,i,ue,ie,g = get_lgbm_train_input(train_positive, negative)

#X=(ue,ie)
X = np.concatenate((ue, ie),axis = 1)

y=g
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
model.score(X_test, y_test)
#params = {    
#          'boosting_type': 'gbdt',
#          'objective': 'binary',
#          'metric': 'auc',
#          'nthread':4,
#          'learning_rate':0.1,
#          'num_leaves':30, 
#          'max_depth': -1,   
#          'subsample': 0.8, 
#          'colsample_bytree': 0.8, 
#    }
#data_train = lgb.Dataset(X_train, y_train)
#cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',early_stopping_rounds=50,seed=0)
#print('best n_estimators:', len(cv_results['auc-mean']))
#print('best cv score:', pd.Series(cv_results['auc-mean']).max())

