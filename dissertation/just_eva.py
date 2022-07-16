# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 04:09:52 2021

@author: stanley
"""

import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
from scipy.spatial import distance
import math
from collections import defaultdict
from args import parser

opts = parser()
def get_user_item_embedding():
    
    #difine user_embd_path and item_embd_path
    user_embd_path = os.path.join(opts.DATASET_DIR, opts.CONCATENATE_DIR, opts.U_EMBD_FILE)
    item_embd_path = os.path.join(opts.DATASET_DIR, opts.CONCATENATE_DIR, opts.I_EMBD_FILE)

    #get user_embd and item_embd
    #user_embd = np.genfromtxt(user_embd_path, dtype=float)
    #item_embd = np.genfromtxt(item_embd_path, dtype=float)
    user_embd = np.load(user_embd_path)
    item_embd = np.load(item_embd_path)
    #get matrix shape
    user_len = len(user_embd)
    item_len = len(item_embd)
    
    return user_embd, item_embd

def get_label():
    path = os.path.join(opts.DATASET_DIR, opts.GT_FILE)
    gt_file = np.genfromtxt(path, dtype='int', delimiter = ',')
    test_path = os.path.join(opts.DATASET_DIR, opts.TEST_DIR, opts.TEST_LIST_FILE)
    test_file = open(test_path, "r")

    test_data=[[] for _ in range(len(gt_file))]
    for idx,line in enumerate(test_file.readlines()):
        iids = line.strip('[').strip('\n').strip(']').strip("'").split(',')
        for iid in iids:
            test_data[idx].append(int(iid))
    label_data = [[] for _ in range(len(gt_file))]
    for idx,user in enumerate(test_data):
        for iid in test_data[idx]:
            if iid in gt_file[idx]:
                label_data[idx].append(1)
            else:
                label_data[idx].append(0)
    return test_data, label_data

def get_ml_label():
    path = os.path.join(opts.DATASET_DIR, opts.GT_FILE)
    gt = np.genfromtxt(path, dtype='int')
    gt_file=[[] for _ in range(943)]
    for idx,data in enumerate(gt):
        gt_file[data[0]].append(data[1])
    test_path = os.path.join(opts.DATASET_DIR, opts.TEST_DIR, opts.TEST_LIST_FILE)
    test_file = open(test_path, "r")

    test_data=[[] for _ in range(len(gt_file))]
    for idx,line in enumerate(test_file.readlines()):
        iids = line.strip('[').strip('\n').strip(']').strip("'").split(',')
        for iid in iids:
            test_data[idx].append(int(iid))
    label_data = [[] for _ in range(len(gt_file))]
    for idx,user in enumerate(test_data):
        for iid in test_data[idx]:
            if iid in gt_file[idx]:
                label_data[idx].append(1)
            else:
                label_data[idx].append(0)
    return test_data, label_data


def triplet_distance_calculation(user_vec, item_vec,test_data, label_data):
    topk_list = range(5,101,5)
    Recall = {}
    Precision = {}
    Hr = {}
    Arhr = {}
    nDCG = {}
    HitRatio = {}
    for k in topk_list:
        Recall[k] = 0
        Precision[k] = 0
        Hr[k] = 0
        Arhr[k] = 0
        nDCG[k] = 0
        HitRatio[k] = 0
    all_dists = []
    Inner = []
    Rank_Index = []
    
    for idx, test_id_list in enumerate(test_data):
        #
#        GT_data = test_id_list[:7]
#        NEG_data = random.sample(test_id_list[7:], 100)
#        one_u_test_data = GT_data + NEG_data
        #
        user_id = idx
        current_user_embd = user_vec[user_id]
        dists = []
        inners = []
        for iid in test_id_list:
            item_id = iid
            current_item_embd = item_vec[item_id]
            
            dist = distance.euclidean(current_user_embd, current_item_embd)
            dists.append(dist)
            inner = np.dot(current_user_embd,np.transpose(current_item_embd))
            inners.append(inner)
        rank_idx = np.argsort(dists)
        rank_label = np.array(label_data[user_id])[rank_idx]
        all_dists.append(dists)
        inner_ra = np.argsort(inners)
        Inner.append(inner_ra)
        Rank_Index.append(rank_idx)
        for k in topk_list:
            ndcg = 0
            HR = 0
            ARHR = 0
            TP = np.sum(rank_label[:k])
            #TN = k-TP
            FP = k-TP
            FN = np.sum(rank_label[k:])
            
            for post, is_hit in enumerate(rank_label[0:k]):
                if is_hit == 1:
                    ndcg += math.log(2, 2) / math.log(post+2, 2)
                    HR +=1
                    break
            for post, is_hit in enumerate(rank_label[0:k]):
                if is_hit == 1:
                    ARHR += 1/(post+1)
                    
            #ML:7, YELP:5, AMAZON:3
            recall = TP/opts.num_gt
            precision = TP/k
            hit_at_k = TP
            Recall[k] = Recall[k] + recall
            Precision[k] = Precision[k] + precision
            Hr[k] = Hr[k] + HR
            Arhr[k] = Arhr[k] + ARHR
            HitRatio[k] = HitRatio[k] + hit_at_k
            nDCG[k] = nDCG[k] + ndcg
        #print(Precision)
    for k in topk_list:
        Recall[k] = Recall[k]/len(label_data)
        Precision[k] = Precision[k]/len(label_data)
        Hr[k] = Hr[k]/len(label_data)
        Arhr[k] = Arhr[k]/ len(label_data)
        
        HitRatio[k] = HitRatio[k]/100#(len(label)-2)
        nDCG[k] = nDCG[k]/100#(len(label)-2)
        
    #return HitRatio, nDCG, Inner, Rank_Index
    return Recall, Precision, Hr, Arhr

user_vec, item_vec = get_user_item_embedding()
test_data, label_data = get_ml_label()
Recall, Precision, Hr, Arhr = triplet_distance_calculation(user_vec, item_vec,test_data, label_data)