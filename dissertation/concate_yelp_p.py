# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 03:44:19 2020

@author: stanley
"""
import numpy as np
import os
DATASET_DIR = 'pcity_data/'

CIGAR_G_PATH = 'yelp_p_CIGAR_global/'
CIGAR_L_PATH = 'yelp_p_CIGAR_local_2020_11_12_Pcity_BPR_Local/'
TRIP_G_PATH = 'yelp_p_trip_global/'
TRIP_L_PATH = 'yelp_p_trip_local/'

TRIP_G_U = 'yelp_pcity_user_emb_64_global.npy'
TRIP_G_I = 'yelp_pcity_item_emb_64_global.npy'
TRIP_L_U = 'yelp_pcity_user_emb_64_5c_local.npy'
TRIP_L_I = 'yelp_pcity_item_emb_64_5c_local.npy'

CIGAR_G_U = 'cigar_userEmb_yelp_p_64.txt'
CIGAR_G_I = 'cigar_itemEmb_yelp_p_64.txt'
CIGAR_L_U = 'yelp_pcity_userEmb_64_BPR_local_kero.txt'
CIGAR_L_I = 'yelp_pcity_itemEmb_64_BPR_local_kero.txt'



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


user_concate_from_row = np.concatenate((t_l_u_emb, c_l_u_emb),axis = 1)
item_concate_from_row = np.concatenate((t_l_i_emb, c_l_i_emb),axis = 1)
#
np.save(os.path.join(DATASET_DIR, 'user_Local_YELPP.npy'), np.array(user_concate_from_row))
np.save(os.path.join(DATASET_DIR, 'item_Local_YELPP.npy'), np.array(item_concate_from_row))

#np.save(os.path.join(DATASET_DIR, 't_g_u_emb.npy'), np.array(t_g_u_emb))
#np.save(os.path.join(DATASET_DIR, 't_g_i_emb.npy'), np.array(t_g_i_emb))
#np.save(os.path.join(DATASET_DIR, 't_l_u_emb.npy'), np.array(t_l_u_emb))
#np.save(os.path.join(DATASET_DIR, 't_l_i_emb.npy'), np.array(t_l_i_emb))
#
#np.save(os.path.join(DATASET_DIR, 'c_g_u_emb.npy'), np.array(c_g_u_emb))
#np.save(os.path.join(DATASET_DIR, 'c_g_i_emb.npy'), np.array(c_g_i_emb))
#np.save(os.path.join(DATASET_DIR, 'c_l_u_emb.npy'), np.array(c_l_u_emb))
#np.save(os.path.join(DATASET_DIR, 'c_l_i_emb.npy'), np.array(c_l_i_emb))
