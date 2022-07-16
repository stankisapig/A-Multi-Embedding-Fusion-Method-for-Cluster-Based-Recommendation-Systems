# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 18:06:08 2020

@author: stanley
"""

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
from data import load_data, get_embedding_input, eva
from args import parser
opts = parser()

usernum, itemnum, train_positive, negative, get_gt, user_test = load_data()

#user_emb, positive_emb, negative_emb= get_embedding_input(train_positive, negative)
#uid ,iid, user_emb, item_emb, gt = get_embedding_input(train_positive, negative)
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

u_cnn_net = CNNNet()
i_cnn_net = CNNNet()
train_loss = tf.keras.metrics.Mean()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00017, epsilon=1e-08)

def train_step(uid, pid, nid):
    with tf.GradientTape() as tape:
        u_emb = u_cnn_net(uid)
        p_emb = i_cnn_net(pid)
        n_emb = i_cnn_net(nid)
        loss = TripletLoss(u_emb, p_emb, n_emb, margin=0.5)
        
    train_loss.update_state(loss)
    triplet_loss = tape.gradient(loss, u_cnn_net.trainable_variables +\
                                        i_cnn_net.trainable_variables)
    optimizer.apply_gradients(zip(triplet_loss, u_cnn_net.trainable_variables +\
                                                i_cnn_net.trainable_variables))

def train(test_iid):
    for epoch in range(opts.num_epochs):
        user_emb, positive_emb, negative_emb= get_embedding_input(train_positive, negative)
        train_dataset = tf.data.Dataset.from_tensor_slices((user_emb, positive_emb, negative_emb))
        train_dataset = train_dataset.shuffle(300)
        train_dataset = train_dataset.batch(1024)
        for user_emb, positive_emb, negative_emb in tqdm(train_dataset):
            train_step(user_emb, positive_emb, negative_emb)
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
                u_emb = u_cnn_net(uid).numpy()
                i_emb = i_cnn_net(item).numpy()
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
            train_loss.reset_states()
    print('Finish Training!')
    
test_iid = np.load(opts.DATASET_DIR+opts.TEST_FILE, allow_pickle=True)
train(test_iid)

user_ids = tf.data.Dataset.range(usernum)
user_ids = user_ids.batch(512)

item_ids = tf.data.Dataset.range(itemnum)
item_ids = item_ids.batch(512)

def GetEmb(user_ids, item_ids):
    user_emb = []
    item_emb = []
    for uid in user_ids:
        #user_emb.append(tf.squeeze(user_net(uid)).numpy())
        uid = np.expand_dims(uid, axis=1)
        user_emb.append(tf.squeeze(u_cnn_net(uid)).numpy())
        #user_emb.append(tf.squeeze(user_net(uid = np.expand_dims(uid, axis=1))).numpy())
    for iid in item_ids:
        iid = np.expand_dims(iid, axis=1)
        item_emb.append(tf.squeeze(i_cnn_net(iid)).numpy())
    user_emb = np.concatenate(user_emb, axis=0)
    item_emb = np.concatenate(item_emb, axis=0)
    return user_emb, item_emb
user_emb, item_emb = GetEmb(user_ids, item_ids)

print(user_emb.shape)
print(item_emb.shape)

#os.makedirs('emb', exist_ok=True)
#dataset = opts.dataset
#np.save(os.path.join(dataset, dataset+'_user_emb_200_50c10_local.npy'), user_emb)
#np.save(os.path.join(dataset, dataset+'_item_emb_200_50c10_local.npy'), item_emb)










