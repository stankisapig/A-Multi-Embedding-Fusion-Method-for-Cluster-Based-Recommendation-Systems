# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:02:00 2020

@author: stanley
"""

import os
import math
import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
import time
from argss import parser

opts = parser()


def eva(u_emb, i_emb, top_k,num_gt):
    is_hit = False
    HR = 0.0
    Recall = 0.0
    Precision = 0.0
    ARHR = 0.0
    TP=0
    dist_table = np.sum(np.square(u_emb-i_emb), axis=1)
    dist_table = np.argsort(dist_table)
    for rank in range(top_k):
        if dist_table[rank] in range(num_gt):
            is_hit = True
            HR += 1
            break
    for rank in range(top_k):
        if dist_table[rank] in range(num_gt):
            TP+=1
            ARHR += 1/(rank+1)
            
    Recall=TP/num_gt
    Precision=TP/top_k
    return Recall, Precision, HR, ARHR, is_hit

def get_train_pos(interaction_path, gt_path):
    data = np.genfromtxt(interaction_path, delimiter='\t')
    test_item = np.genfromtxt(gt_path, dtype=int, delimiter=',')
    all_user = data[:,0]
    all_item = data[:,1]
    all_rate = data[:,2]

    num_user = len(set(all_user))
    num_item = len(set(all_item))
    
    # 從0開始編號
    all_item_set = [[] for _ in range(num_user)]
    for idx in range(all_rate.shape[0]):
        if interaction_path[:2] == 'ml':
            all_item_set[int(all_user[idx])].append(int(all_item[idx]))
        else:
            all_item_set[int(all_user[idx])].append(int(all_item[idx]))
    
    
    train_pos = []
    for now_user in range(num_user):
        current_positive = np.setdiff1d(all_item_set[now_user],test_item[now_user])
        train_pos.append(current_positive)

    return train_pos, num_user, num_item

def load_negative(path):
    negative_item = []
    for idx, line in enumerate(open(path)):
        cur = np.array(line.strip().strip('[]').split(', ')).astype(np.uint16)
        negative_item.append(cur)
    return negative_item
        

def get_input(train_pos, negative_item):
    user = []
    p_item = []
    n_item = []
    for uid in range(len(train_pos)):
        if not train_pos[uid] == []:
            random_id = np.random.choice(negative_item[uid].shape[0], 
                                         train_pos[uid].shape[0])
            user.append(np.array([uid]*train_pos[uid].shape[0]))
            p_item.append(train_pos[uid])
            n_item.append(negative_item[uid][random_id])
    user = np.concatenate(user)
    p_item = np.concatenate(p_item)
    n_item = np.concatenate(n_item)
    
    return np.expand_dims(user, axis=1), np.expand_dims(p_item, axis=1), np.expand_dims(n_item, axis=1)