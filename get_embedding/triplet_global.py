# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 02:27:29 2020

@author: stanley
"""

import os
from data_global import get_train_pos, get_input, load_negative, eva
import numpy as np
from time import sleep
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Lambda,
    Flatten,
)
from argss import parser

opts = parser()
#path = os.path.join('ml_100k', 'u.data')
interaction_path = os.path.join(opts.dataset,'interaction', opts.interaction_file)
test_data_path = os.path.join('test_data',opts.test_file)
gt_path = os.path.join(opts.dataset,'interaction', opts.gt_file)
train_pos,negative_item ,num_user, num_item = get_train_pos(interaction_path, gt_path)
def UserNet():
    x = inputs = Input([1], name='input')
    x =  Flatten()(Embedding(num_user, opts.embd_dim)(x))
    output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    return Model(inputs, output, name='user_net')

def ItemNet():
    x = inputs = Input([1], name='input')
    x =  Flatten()(Embedding(num_item, opts.embd_dim)(x))
    output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    return Model(inputs, output, name='item_net')

def TripletLoss(u_emb, p_emb, n_emb, margin=0.5):
    pos_dist = tf.reduce_sum(tf.math.square(u_emb - p_emb), axis=1)
    neg_dist = tf.reduce_sum(tf.math.square(u_emb - n_emb), axis=1)
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.keras.activations.relu(basic_loss)
    return loss

user_net = UserNet()
item_net = ItemNet()
train_loss = tf.keras.metrics.Mean()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00017, epsilon=1e-08)

def train_step(uid, pid, nid):
    with tf.GradientTape() as tape:
        u_emb = user_net(uid)
        p_emb = item_net(pid)
        n_emb = item_net(nid)
        loss = TripletLoss(u_emb, p_emb, n_emb, margin=0.5)
        
    train_loss.update_state(loss)
    triplet_loss = tape.gradient(loss, user_net.trainable_variables +\
                                        item_net.trainable_variables)
    optimizer.apply_gradients(zip(triplet_loss, user_net.trainable_variables +\
                                                item_net.trainable_variables))

def train(test_iid):
    best_loss = 1000
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
                u_emb = user_net(uid).numpy()
                i_emb = item_net(item).numpy()
                recall, precision, hr, arhr, is_hit = eva(u_emb, i_emb, opts.topk, opts.num_gt)
                if is_hit:
                    Recall += recall
                    Precision += precision
                    Hr += 1
                    Arhr += arhr
            sleep(0.2)
            print(float(Recall/len(test_iid)))
            print(float(Precision/len(test_iid)))
            print(float(Hr/len(test_iid)))
            print(float(Arhr/len(test_iid)))
            sleep(0.2)
            
    #        if train_loss.result().numpy() < best_loss:
    #            best_loss = train_loss.result().numpy()
    #            stop_count = 0
    #        else:
    #            stop_count = stop_count + 1
    #        if stop_count == 3:
    #            break
            
            train_loss.reset_states()
    print('Finish Training!')

test_path = test_data_path
#test_path = os.path.join(opts.dataset, opts.dataset+'_test_data.npy')
test_iid = np.load(test_path, allow_pickle=True)
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
        user_emb.append(tf.squeeze(user_net(uid)).numpy())
        #user_emb.append(tf.squeeze(user_net(uid = np.expand_dims(uid, axis=1))).numpy())
    for iid in item_ids:
        iid = np.expand_dims(iid, axis=1)
        item_emb.append(tf.squeeze(item_net(iid)).numpy())
    user_emb = np.concatenate(user_emb, axis=0)
    item_emb = np.concatenate(item_emb, axis=0)
    return user_emb, item_emb
user_emb, item_emb = GetEmb(user_ids, item_ids)

print(user_emb.shape)
print(item_emb.shape)

#os.makedirs('emb', exist_ok=True)
dataset = opts.dataset
np.save(os.path.join(dataset, dataset+'_user_emb_200_global.npy'), user_emb)
np.save(os.path.join(dataset, dataset+'_item_emb_200_global.npy'), item_emb)