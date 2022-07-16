# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 05:55:16 2021

@author: stanley
"""

import numpy as np
import os
DATASET_DIR = 'ml_data/'
CONCATENATE_DIR = 'Concatenate/'

CIGAR_G_PATH = 'GC/'
CIGAR_L_PATH = 'LC/'
TRIP_G_PATH = 'GE/'
TRIP_L_PATH = 'LE/'

TRIP_G_U = 'CEU.npy'
TRIP_G_I = 'CEI.npy'
TRIP_L_U = 'LEU.npy'
TRIP_L_I = 'LEI.npy'

CIGAR_G_U = 'GCU.txt'
CIGAR_G_I = 'GCI.txt'
CIGAR_L_U = 'LCU.txt'
CIGAR_L_I = 'LCI.txt'

T_G_PATH = DATASET_DIR + TRIP_G_PATH
T_L_PATH = DATASET_DIR + TRIP_L_PATH
T_G_U_EMB_PATH = T_G_PATH+ TRIP_G_U
T_G_I_EMB_PATH = T_G_PATH + TRIP_G_I
T_L_U_EMB_PATH = T_L_PATH + TRIP_L_U
T_L_I_EMB_PATH = T_L_PATH + TRIP_L_I


T_G_PATH = DATASET_DIR + CIGAR_G_PATH
T_L_PATH = DATASET_DIR + CIGAR_L_PATH
C_G_U_EMB_PATH = T_G_PATH + CIGAR_G_U
C_G_I_EMB_PATH = T_G_PATH + CIGAR_G_I
C_L_U_EMB_PATH = T_L_PATH + CIGAR_L_U
C_L_I_EMB_PATH = T_L_PATH + CIGAR_L_I

t_g_u_emb = np.load(T_G_U_EMB_PATH)
t_g_i_emb = np.load(T_G_I_EMB_PATH)
t_l_u_emb = np.load(T_L_U_EMB_PATH)
t_l_i_emb = np.load(T_L_I_EMB_PATH)

c_g_u_emb = np.genfromtxt(C_G_U_EMB_PATH)
c_g_i_emb = np.genfromtxt(C_G_I_EMB_PATH)
c_l_u_emb = np.genfromtxt(C_L_U_EMB_PATH)
c_l_i_emb = np.genfromtxt(C_L_I_EMB_PATH)



def get_norm_emb(type_emb):
    final_embedding =[]
    for emb in type_emb:
        new_emb = []
        nurm_val = 0
        for val in emb:
            sqr_val = val**2
            nurm_val += sqr_val
        nurm_val = np.sqrt(nurm_val)
        for val in emb:
            new_val= val/nurm_val
            new_emb.append(new_val)
        final_embedding.append(new_emb)
    final_embedding = np.array(final_embedding)
    
    return final_embedding


t_g_u_emb = get_norm_emb(t_g_u_emb)
t_g_i_emb = get_norm_emb(t_g_i_emb)
t_l_u_emb = get_norm_emb(t_l_u_emb)
t_l_i_emb = get_norm_emb(t_l_i_emb)

c_g_u_emb = get_norm_emb(c_g_u_emb)
c_g_i_emb = get_norm_emb(c_g_i_emb)
c_l_u_emb = get_norm_emb(c_l_u_emb)
c_l_i_emb = get_norm_emb(c_l_i_emb)

user_concate_from_row = np.concatenate((t_g_u_emb, t_l_u_emb, c_g_u_emb, c_l_u_emb),axis = 1)
item_concate_from_row = np.concatenate((t_g_i_emb, t_l_i_emb, c_g_i_emb, c_l_i_emb),axis = 1)
#
#np.save(os.path.join(DATASET_DIR, CONCATENATE_DIR, 'user_concate_ML.npy'), np.array(user_concate_from_row))
#np.save(os.path.join(DATASET_DIR, CONCATENATE_DIR, 'item_concate_ML.npy'), np.array(item_concate_from_row))

user_concate_without_LC = np.concatenate((t_l_u_emb, c_l_u_emb),axis = 1)
item_concate_without_LC = np.concatenate((t_l_i_emb, c_l_i_emb),axis = 1)
#
np.save(os.path.join(DATASET_DIR, CONCATENATE_DIR, 'user_Local_ML.npy'), np.array(user_concate_without_LC))
np.save(os.path.join(DATASET_DIR, CONCATENATE_DIR, 'item_Local_ML.npy'), np.array(item_concate_without_LC))


#np.save(os.path.join(DATASET_DIR, CONCATENATE_DIR, 't_g_u_emb_ML.npy'), np.array(t_g_u_emb))
#np.save(os.path.join(DATASET_DIR, CONCATENATE_DIR, 't_g_i_emb_ML.npy'), np.array(t_g_i_emb))
#np.save(os.path.join(DATASET_DIR, CONCATENATE_DIR, 't_l_u_emb_ML.npy'), np.array(t_l_u_emb))
#np.save(os.path.join(DATASET_DIR, CONCATENATE_DIR, 't_l_i_emb_ML.npy'), np.array(t_l_i_emb))
#
#np.save(os.path.join(DATASET_DIR, CONCATENATE_DIR, 'c_g_u_emb_ML.npy'), np.array(c_g_u_emb))
#np.save(os.path.join(DATASET_DIR, CONCATENATE_DIR, 'c_g_i_emb_ML.npy'), np.array(c_g_i_emb))
#np.save(os.path.join(DATASET_DIR, CONCATENATE_DIR, 'c_l_u_emb_ML.npy'), np.array(c_l_u_emb))
#np.save(os.path.join(DATASET_DIR, CONCATENATE_DIR, 'c_l_i_emb_ML.npy'), np.array(c_l_i_emb))
