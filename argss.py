# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:30:02 2020

@author: stanley
"""

import os

class parser(object):
    def __init__(self):
        self.dataset = 'yelp_pcity'
        self.interaction_file = 'new_yelp_r8_interaction_reid.txt'
        self.test_file = 'yelp_p_test_5gt.pkl'
        self.gt_file = 'P-YELP-5GT.txt'
        self.negative_file = 'YELP_Pcity_5clusters_negItems.txt'
        self.embd_dim=200
        self.num_epochs = 200
        self.topk = 10
        self.num_gt = 5
        