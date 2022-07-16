# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:41:19 2020

@author: stanley
"""
import os
import numpy as np
from collections import defaultdict
from args import parser
opts = parser()

path = os.path.join(opts.DATASET_DIR, opts.TEST_DIR, opts.TEST_LIST_FILE)
test_file = open(path, "r")

#build test_list_dict (contain train positive)
test_list_dict = defaultdict(list)
for idx,line in enumerate(test_file.readlines()):
    iids = line.strip('[').strip('\n').strip(']').strip("'").split(',')
    for iid in iids:
        test_list_dict[idx].append(int(iid))