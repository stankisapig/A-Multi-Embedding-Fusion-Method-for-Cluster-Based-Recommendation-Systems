# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 03:32:09 2020

@author: stanley
"""

import os

class parser(object):
    def __init__(self):
        '''-------------------------------Data Setting-------------------------------'''
        #pcity settings
#        self.dataset = 'yelp_p'
#        self.DATASET_DIR = 'pcity_data/'
#        self.CONCATENATE_DIR = 'Concatenate/'
#        self.ALL_TEST = 'all_test/'
#        self.TEST_DIR = '2020_11_12_Pcity_Testset/10_clusters/'
#        
#        self.INTERACTIONS_FILE = 'new_yelp_r8_interaction_reid.txt'
#        self.TEST_LIST_FILE = 'pcity_testset_C10_Top2.txt'
#        self.GT_FILE = 'P-YELP-5GT.txt'
#        self.TEST_FILE = 'yelp_p_test_5gt.pkl'
#        self.U_EMBD_FILE = 'user_Local_YELPP.npy'
#        self.I_EMBD_FILE = 'item_Local_YELPP.npy'
#        self.NEGATIVE_FILE = 'YELP_Pcity_5clusters_negItems.txt'
#        self.topk = 5
#        self.num_epochs = 300
#        self.num_gt = 5
#        self.lr = 0.00017
        #mcity settings
#        self.dataset = 'yelp_m'
#        self.DATASET_DIR = 'mcity_data/'
#        self.CONCATENATE_DIR = 'Concatenate/'
#        self.TEST_DIR = '2020_11_14_Mcity_Testset/10_clusters/'
#        
#        self.INTERACTIONS_FILE = 'yelp_mcity_new_interactions_reid.txt'
#        self.TEST_LIST_FILE = 'mcity_testset_C10_Top2.txt'
#        self.GT_FILE = 'M-YELP-5GT.txt'
#        self.TEST_FILE = 'yelp_m_test_5gt.pkl'
#        self.U_EMBD_FILE = 't_g_u_emb_YELPM.npy'
#        self.I_EMBD_FILE = 't_g_i_emb_YELPM.npy'
#        self.NEGATIVE_FILE = 'YELP_Mcity_C10_5clusters_neg.txt'
#        self.topk = 5
#        self.num_epochs = 400
#        self.num_gt = 5
#        self.lr = 0.00017
        
        
        #ML settings
        self.dataset = 'ml'
        self.DATASET_DIR = 'ml_data/'
        self.CONCATENATE_DIR = 'Concatenate/'
        self.TEST_DIR = '2020_12_17_ML_100k_Testset/'
        
        self.INTERACTIONS_FILE = 'ml_100k_interaction.txt'
        self.TEST_LIST_FILE = 'ml_100k_testset_C10_Top2.txt'
        self.GT_FILE = 'ml_100k_7GT.txt'
        self.TEST_FILE = 'ml_test_7gt.pkl'
        self.U_EMBD_FILE = 't_g_u_emb_ML.npy'
        self.I_EMBD_FILE = 't_g_i_emb_ML.npy'
        #self.NEGATIVE_FILE = 'YELP_Mcity_C10_5clusters_neg.txt'
        self.topk = 5
        self.num_epochs = 300
        self.num_gt = 7
        self.lr = 0.00017
        
        
        