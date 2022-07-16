# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:22:04 2020

@author: stanley
"""
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import sleep
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Lambda,
    Flatten,
)
from data import eva, get_train_pos, get_input, index_embedding, gen_negative
from args import parser
opts = parser()


interaction_path = os.path.join(opts.DATASET_DIR, opts.INTERACTIONS_FILE)
#test_data_path = os.path.join('test_data',opts.test_file)
gt_path = os.path.join(opts.DATASET_DIR, opts.GT_FILE)
negative_path = os.path.join(opts.DATASET_DIR, opts.NEGATIVE_FILE)
train_pos, num_user, num_item = get_train_pos(interaction_path, gt_path)
#negative_item = load_negative(negative_path)
negative_item = gen_negative()

def CNNNet():
    x = inputs = Input([256], name='input')
    x = Dense(128, activation='sigmoid')(x)
    hidden2 = Dense(64, activation='sigmoid')(x)
    #output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    return Model(inputs, hidden2, name='user_net')

def TripletLoss(u_emb, p_emb, n_emb, margin=0.5):
    pos_dist = tf.reduce_sum(tf.math.square(u_emb - p_emb), axis=1)
    neg_dist = tf.reduce_sum(tf.math.square(u_emb - n_emb), axis=1)
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.keras.activations.relu(basic_loss)
    return loss


user_net = CNNNet()
item_net = CNNNet()
train_loss = tf.keras.metrics.Mean()
optimizer = tf.keras.optimizers.Adam(learning_rate=opts.lr, epsilon=1e-08)

def train_step(uid, pid, nid):
    ue, pe, ne = index_embedding(uid, pid, nid)
#    ue = tf.squeeze(ue).numpy()
#    pe = tf.squeeze(pe).numpy()
#    ne = tf.squeeze(ne).numpy()
    with tf.GradientTape() as tape:
        u_emb = user_net(ue)
        p_emb = item_net(pe)
        n_emb = item_net(ne)
        loss = TripletLoss(u_emb, p_emb, n_emb, margin=0.5)
        
    train_loss.update_state(loss)
    triplet_loss = tape.gradient(loss, user_net.trainable_variables +\
                                        item_net.trainable_variables)
    optimizer.apply_gradients(zip(triplet_loss, user_net.trainable_variables +\
                                                item_net.trainable_variables))

def train(test_iid):
    top_Recall = 0
    top_Precision = 0
    top_Hr = 0
    top_Arhr = 0
    for epoch in range(opts.num_epochs):
        uid, pid, nid = get_input(train_pos, negative_item)
        train_dataset = tf.data.Dataset.from_tensor_slices((uid, pid, nid))
        train_dataset = train_dataset.shuffle(300)
        train_dataset = train_dataset.batch(1024)
        for uid, pid, nid in tqdm(train_dataset):
            train_step(uid, pid, nid)
        sleep(0.2)
        print('Epoch :{}, Loss :{}'.format(epoch+1, train_loss.result().numpy()))
        sleep(0.2)
            
        Recall = 0
        Precision = 0
        Hr = 0
        Arhr = 0

        if (epoch % 10==0):
            for uid in tqdm(range(len(test_iid))):
                item = np.expand_dims(np.array(test_iid[uid]), axis=1)
                uid = np.expand_dims(np.array([int(uid)]), axis=1)
                #item = np.array(test_iid[uid])
                #uid = np.array([int(uid)])
                uid_x,item_x,item_xx= index_embedding(uid,item,item)
                uid_x = user_net(uid_x).numpy()
                item_x = item_net(item_x).numpy()
                #uid_x = tf.squeeze(user_net(uid_x),axis=1).numpy()
                #item_x = tf.squeeze(user_net(item_x)).numpy()
                recall, precision, hr, arhr, is_hit = eva(uid_x, item_x, opts.topk, opts.num_gt)
                if is_hit:
                    Recall += recall
                    Precision += precision
                    Hr += 1
                    Arhr += arhr
            sleep(0.2)
            final_recall = float(Recall/len(test_iid))
            final_precision = float(Precision/len(test_iid))
            final_hr = float(Hr/len(test_iid))
            final_arhr = float(Arhr/len(test_iid))
            print(final_recall)
            print(final_precision)
            print(final_hr)
            print(final_arhr)
            sleep(0.2)
            train_loss.reset_states()
            if final_recall> top_Recall:
                top_Recall=final_recall
            if final_precision> top_Precision:
                top_Precision=final_precision
            if final_hr> top_Hr:
                top_Hr=final_hr
            if final_arhr> top_Arhr:
                top_Arhr=final_arhr
    print('top_Recall    : ',top_Recall)
    print('top_Precision : ',top_Precision)
    print('top_Hr        : ',top_Hr)
    print('top_Arhr      : ',top_Arhr)
                
    print('Finish Training!')
    
test_iid = np.load(opts.DATASET_DIR+opts.TEST_FILE, allow_pickle=True)
train(test_iid)

user_ids = tf.data.Dataset.range(num_user)
user_ids = user_ids.batch(512)

item_ids = tf.data.Dataset.range(num_item)
item_ids = item_ids.batch(512)

def GetEmb(user_ids, item_ids):
    user_emb = []
    item_emb = []
    for uid in user_ids:
        #user_emb.append(tf.squeeze(user_net(uid)).numpy())
        uid = np.expand_dims(uid, axis=1)
        uid_a,uid_aa,uid_aaa= index_embedding(uid,uid,uid)
        user_emb.append(tf.squeeze(user_net(uid_a)).numpy())
        #user_emb.append(tf.squeeze(user_net(uid = np.expand_dims(uid, axis=1))).numpy())
    for iid in item_ids:
        iid = np.expand_dims(iid, axis=1)
        iid_a,iid_aa,iid_aaa= index_embedding(iid*0,iid,iid)
        item_emb.append(tf.squeeze(item_net(iid_aa)).numpy())
    user_emb = np.concatenate(user_emb, axis=0)
    item_emb = np.concatenate(item_emb, axis=0)
    return user_emb, item_emb
user_emb, item_emb = GetEmb(user_ids, item_ids)

print(user_emb.shape)
print(item_emb.shape)

#os.makedirs('emb', exist_ok=True)
dataset = opts.dataset
#np.save(os.path.join(opts.DATASET_DIR, dataset+'_user_ensenble_embedding.npy'), user_emb)
#np.save(os.path.join(opts.DATASET_DIR, dataset+'_item_ensenble_embedding.npy'), item_emb)




















