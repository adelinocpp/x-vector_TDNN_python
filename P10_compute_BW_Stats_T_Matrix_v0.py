#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 09:34:01 2021

@author: adelino
"""
DEBUG_MODE = False
if (DEBUG_MODE):
    import sys
import configure as c
import pickle
from DB_wav_reader import find_feats
import os
from utils.compute_bw_stats import compute_bw_stats
from utils.train_tv_space import train_tv_space
import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------        
Compute_BW_stats_T_matrix = True
if (not os.path.exists(c.TRAIN_BNF_DIR)):
    print("Diretório de características não existe. Executar a rotina P01.")
    Compute_BW_stats_T_matrix = False        
    
if (not os.path.exists(c.TRAIN_BNF_DIR)):
    print("Diretório de características não existe. Executar a rotina P01.")
    Compute_BW_stats_T_matrix = False
    
if (Compute_BW_stats_T_matrix):
    with open(c.GMM_UBM_FILE_NAME, 'rb') as f:
        UBM = pickle.load(f)
    
    n_max_iter = 200
    tv_dim = 1800
    
    file_list = find_feats(c.TRAIN_FEAT_DIR)
    file_list.sort()
    
    splitDir = c.IVECTOR_TRAIN_DIR.split('/')
    for idx in range(1,len(splitDir)+1):
        curDir = '/'.join(splitDir[:idx])
        if (not os.path.exists(curDir)):
            os.mkdir(curDir)
    
    # nStop = 3
    # statsTrain = np.zeros((nStop,UBM.n_components*(UBM.n_features_in_ + 1) ) )
    statsTrain = np.zeros((len(file_list),UBM.n_components*(UBM.n_features_in_ + 1) ) )
    
    for idx, filename in enumerate(file_list):
        with open(filename, 'rb') as f:
            feat_and_label = pickle.load(f)
        
        # filenameParts = filename.replace('\\', '/')
        # filenameFolder = filenameParts.split('/')[-2]
        # filenameBase = filenameParts.split('/')[-1].split('.')[0]
        # filenameSave = c.IVECTOR_TRAIN_DIR + '/' + filenameFolder + '/' + filenameBase + '.p'            
        N, F = compute_bw_stats(feat_and_label['feat'], UBM);
        statsTrain[idx,:] = np.append(N,F)
        
        # if (not os.path.exists(c.IVECTOR_TRAIN_DIR + '/' + filenameFolder)):
        #     os.mkdir(c.IVECTOR_TRAIN_DIR + '/' + filenameFolder)
        # with open(filenameSave, 'wb') as f:
        #     pickle.dump(gmm_and_label,f)
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
        # if (idx == (nStop-1)):
        #     tv_dim = 60
        #     break

    # scaler = StandardScaler()
    # statsTrain = scaler.fit_transform(statsTrain)
    # with open(c.BW_SCALER_FILE, 'wb') as f:
    #     pickle.dump(scaler,f)
    # print('Scaler data: mean {:}, var {:}'.format(scaler.mean_.mean(),scaler.var_.mean()))
    
    T_matrix = train_tv_space(statsTrain, UBM, tv_dim, n_max_iter, c.T_MATRIX_FILE_NAME)
        
    # if (DEBUG_MODE):
    #     sys.exit("MODO DEPURACAO: Fim do script")