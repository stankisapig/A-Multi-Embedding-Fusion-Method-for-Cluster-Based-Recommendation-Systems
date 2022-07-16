# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 03:30:30 2020

@author: stanley
"""

import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
import tensorflow as tf
from args import parser
opts = parser()


def load_data():
    #load all interaction file
    path = os.path.join(opts.DATASET_DIR, opts.INTERACTIONS_FILE)
    interaction_data = np.genfromtxt(path,dtype='int')
    usernum = 0
    itemnum = 0
    #build interactions_dict
    interactions_dict = defaultdict(list)
    for interaction in interaction_data:
        uid = interaction[0]
        iid = interaction[1]
        
        usernum = max(uid, usernum)
        itemnum = max(iid, itemnum)
        interactions_dict[uid].append(iid)
        
    # assume user/item index starting from 0
    usernum += 1
    itemnum += 1
    
    #load test list file
    path = os.path.join(opts.DATASET_DIR, opts.TEST_DIR, opts.TEST_LIST_FILE)
    test_file = open(path, "r")
    
    #build test_list_dict (contain train positive)
    test_list_dict = defaultdict(list)
    for idx,line in enumerate(test_file.readlines()):
        iids = line.strip('[').strip('\n').strip(']').strip("'").split(',')
        for iid in iids:
            test_list_dict[idx].append(int(iid))
            
    #load GT file
    path = os.path.join(opts.DATASET_DIR, opts.GT_FILE)
    gt_file = np.genfromtxt(path, dtype='int', delimiter = ',')
    #build gt_dict
    gt_dict = defaultdict(list)
    for idx, n_gt in enumerate(gt_file):
        for gt in n_gt:
            gt_dict[idx].append(gt)
            
    #define train_positive, train_negative, get_gt, user_test, all_item
    train_positive = [[] for _ in range(usernum)]
    train_negative = [[] for _ in range(usernum)]
    get_gt = [[] for _ in range(usernum)]
    user_test = [[] for _ in range(usernum)]
    negative = [[] for _ in range(usernum)]
    all_item = [item for item in range(itemnum)]
    all_item = np.array(all_item)
    for user in interactions_dict:
        #one user all interactions
        oneu_inter = np.array(interactions_dict[user])
        #one user gt
        oneu_gt = np.array(gt_dict[user])
        #get user gt
        get_gt[user] = oneu_gt
        #one user test list (contain train positive)
        oneu_test_list = np.array(test_list_dict[user])
        #get train positive
        oneu_train_positive = np.setdiff1d(oneu_inter,oneu_gt)
        train_positive[user] = oneu_train_positive
        
        #get negative
        negative[user] = np.setdiff1d(all_item,oneu_inter)
        #get test data(test list(contain train positive) - train positive)
        oneu_test_data = np.setdiff1d(oneu_test_list,oneu_train_positive)
        user_test[user] = oneu_test_data
        #one user all train item
        oneu_all_train = np.setdiff1d(all_item,oneu_test_data)
        #get train negative
        oneu_train_negative = np.setdiff1d(oneu_all_train,oneu_train_positive)
        train_negative[user] = oneu_train_negative
        
    return usernum, itemnum, train_positive, negative, get_gt, user_test

def gen_negative():
    #load all interaction file
    path = os.path.join(opts.DATASET_DIR, opts.INTERACTIONS_FILE)
    interaction_data = np.genfromtxt(path,dtype='int')
    usernum = 0
    itemnum = 0
    #build interactions_dict
    interactions_dict = defaultdict(list)
    for interaction in interaction_data:
        uid = interaction[0]
        iid = interaction[1]
        
        usernum = max(uid, usernum)
        itemnum = max(iid, itemnum)
        interactions_dict[uid].append(iid)
        
    # assume user/item index starting from 0
    usernum += 1
    itemnum += 1
    
    all_item = [item for item in range(itemnum)]
    all_item = np.array(all_item)
    negative = [[] for _ in range(usernum)]
    for user in interactions_dict:
        oneu_inter = np.array(interactions_dict[user])
        
        negative[user] = np.setdiff1d(all_item,oneu_inter)
    return negative

def get_embedding_input(train_pos, neg):
    user_embedding = np.load(os.path.join(opts.DATASET_DIR, opts.CONCATENATE_DIR, opts.U_EMBD_FILE))
    item_embedding = np.load(os.path.join(opts.DATASET_DIR, opts.CONCATENATE_DIR, opts.I_EMBD_FILE))
    user = []
    p_item = []
    n_item = []
    user_id = []
    pid = []
    nid = []
    for uid in range(len(train_pos)):
        if not train_pos[uid] == []:
            random_id = np.random.choice(neg[uid].shape[0], 
                                         train_pos[uid].shape[0])
            user.append(user_embedding[np.array([uid]*train_pos[uid].shape[0])])
            p_item.append(item_embedding[train_pos[uid]])
            n_item.append(item_embedding[neg[uid][random_id]])
            user_id.append(np.array([uid]*train_pos[uid].shape[0]))
            pid.append(train_pos[uid])
            nid.append(neg[uid][random_id])
    user = np.concatenate(user)
    p_item = np.concatenate(p_item)
    n_item = np.concatenate(n_item)
    user_id = np.concatenate(user_id)
    pid = np.concatenate(pid)
    nid = np.concatenate(nid)
    return user, p_item, n_item
    #return np.expand_dims(user, axis=1), np.expand_dims(p_item, axis=1), np.expand_dims(n_item, axis=1)
    
def get_lgbm_train_input(train_pos, neg):
    user_embedding_data = np.load(os.path.join(opts.DATASET_DIR, opts.CONCATENATE_DIR, opts.U_EMBD_FILE))
    item_embedding_data = np.load(os.path.join(opts.DATASET_DIR, opts.CONCATENATE_DIR, opts.I_EMBD_FILE))
    user = []
    user_emb = []
    item = []
    item_emb = []
    gt = []
    for uid in range(len(train_pos)):
        if not train_pos[uid] == []:
            user.append(np.array([uid]*train_pos[uid].shape[0]))
            user_emb.append(user_embedding_data[np.array([uid]*train_pos[uid].shape[0])])
            item.append(train_pos[uid])
            item_emb.append(item_embedding_data[train_pos[uid]])
            gt.append(np.array([1]*train_pos[uid].shape[0]))
    for uid in range(len(neg)):
        if not neg[uid] == []:
            user.append(np.array([uid]*neg[uid].shape[0]))
            user_emb.append(user_embedding_data[np.array([uid]*neg[uid].shape[0])])
            item.append(neg[uid])
            item_emb.append(item_embedding_data[neg[uid]])
            gt.append(np.array([0]*neg[uid].shape[0]))
    user = np.concatenate(user)
    item = np.concatenate(item)
    user_emb = np.concatenate(user_emb)
    item_emb = np.concatenate(item_emb)
    gt = np.concatenate(gt)
    
    return np.expand_dims(user, axis=1), np.expand_dims(item, axis=1), user_emb, item_emb, gt

def get_lgbm_train(train_pos, neg):
    user_embedding_data = np.load(os.path.join(opts.DATASET_DIR, opts.CONCATENATE_DIR, opts.U_EMBD_FILE))
    item_embedding_data = np.load(os.path.join(opts.DATASET_DIR, opts.CONCATENATE_DIR, opts.I_EMBD_FILE))
    user_emb = [[] for x in range(len(user_embedding_data))]
    gt = [[] for x in range(len(user_embedding_data))]
    for uid in range(len(train_pos)):
        for items in train_pos[uid]:
            user_emb[uid].append(items)
            gt[uid].append(1)
        for items in neg[uid]:
            user_emb[uid].append(items)
            gt[uid].append(0)
    train_data = []
    train_gt = []
    for uid in range(len(user_emb)):
        u_array = np.array(user_emb[uid])
        train_data.append(u_array)
    for uid in range(len(gt)):
        gt_array = np.array(gt[uid])
        train_gt.append(gt_array)
    
    return train_data, train_gt
    
def get_lgbm_test():
    user_embedding_data = np.load(os.path.join(opts.DATASET_DIR, opts.CONCATENATE_DIR, 'user_concate_YELPM.npy'))
    item_embedding_data = np.load(os.path.join(opts.DATASET_DIR, opts.CONCATENATE_DIR, 'item_concate_YELPM.npy'))
    user_emb = [[] for x in range(len(user_embedding_data))]
    gt = [[] for x in range(len(user_embedding_data))]
    
    
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

def lgbm_eva(dist_table, top_k,num_gt):
    is_hit = False
    Recall = 0.0
    Precision = 0.0
    TP=0
    for rank in range(top_k):
        if dist_table[rank] in range(num_gt):
            TP+=1
            
    Recall=TP/num_gt
    Precision=TP/top_k
    return Recall, Precision

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


def index_embedding(uid, pid, nid):
    user_embedding = np.load(os.path.join(opts.DATASET_DIR, opts.CONCATENATE_DIR, opts.U_EMBD_FILE))
    item_embedding = np.load(os.path.join(opts.DATASET_DIR, opts.CONCATENATE_DIR, opts.I_EMBD_FILE))
    user = user_embedding[uid]
    pitem = item_embedding[pid]
    nitem = item_embedding[nid]

    return tf.squeeze(user,axis=1).numpy(), tf.squeeze(pitem).numpy(), tf.squeeze(nitem)

def lgbm_index_embedding(current_user, current_item):
    user_embedding = np.load(os.path.join(opts.DATASET_DIR, opts.CONCATENATE_DIR, opts.U_EMBD_FILE))
    item_embedding = np.load(os.path.join(opts.DATASET_DIR, opts.CONCATENATE_DIR, opts.I_EMBD_FILE))
    user = user_embedding[current_user]
    item = item_embedding[current_item]
    ui_vector = np.concatenate((user, item),axis = 0)
    #return tf.convert_to_tensor(user).numpy(), tf.convert_to_tensor(item).numpy()
    return ui_vector




