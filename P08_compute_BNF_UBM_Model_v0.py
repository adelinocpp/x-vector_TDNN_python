#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 09:35:41 2021

@author: adelino
"""
DEBUG_MODE = False
import os
import configure as c
from DB_wav_reader import find_wavs, find_feats
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
import math
if (DEBUG_MODE):
    import sys

Compute_GMM_UBM_Train = True
if (not os.path.exists(c.TRAIN_BNF_DIR)):
    print("Diretório de características não existe. Executar a rotina P01.")
    Compute_GMM_UBM_Train = False
        
if (Compute_GMM_UBM_Train):
    file_list = find_feats(c.TRAIN_BNF_DIR)
    file_list.sort()
    print('Inicio da caraga de {} arquivos'.format(len(file_list)))
    for idx, filename in enumerate(file_list):
        # print('Carregando arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
        with open(filename, 'rb') as f:
            feat_and_label = pickle.load(f)
        if (idx == 0):
            ubmData = np.empty(shape=[0, feat_and_label['feat'].shape[1]])
        ubmData = np.append(ubmData,  feat_and_label['feat'],axis=0)
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));

    print("Treinando UBM!")
    nFrames = ubmData.shape[0]
    ds_factor = 4
    ubmIniPow = 0
    if (os.path.exists(c.BNF_GMM_UBM_FILE_NAME)):
        print("Modelo UBM em desenvolvimento, carregando UBM...")
        with open(c.BNF_GMM_UBM_FILE_NAME, 'rb') as f:
            UBM = pickle.load(f)
        print("Carregado UBM com {:} componentes...".format(UBM.n_components))
        ubmIniPow = int(np.log2(UBM.n_components)) + 1;
    else:
        print("Iniciando outro modelo UBM...")
        
    for nPow in range(ubmIniPow,int(1+math.log2(c.nComponents))):
        nComp = 2 ** nPow
        print('Iniciando com {} componentes'.format(nComp))
        if (nComp < 0.5*c.nComponents ):
            idxSel = np.random.permutation(nFrames)[0:int(nFrames/ds_factor)]
            idxSel.sort()
            regData = ubmData[idxSel,:]
        else:
            regData = ubmData
        
        if (nComp == 1):
            UBM = GaussianMixture(n_components = nComp, covariance_type=c.covType, 
                                  reg_covar=1e-6,init_params='kmeans', n_init=1, 
                                  tol=1e-3, verbose = 2, max_iter=200).fit(regData)
        else:
            epsM = np.zeros(UBM.means_.shape)
            idxMaxPrec = np.argmax(UBM.covariances_.max(0))
            epsM[:,idxMaxPrec] = np.sqrt(np.max(UBM.covariances_.max(0)))*np.ones([1,UBM.precisions_.shape[0]])
            wIni = 0.5*np.append(UBM.weights_,UBM.weights_,axis=0)
            mIni = np.append(UBM.means_ - epsM,UBM.means_ + epsM,axis=0)
            pIni = np.append(UBM.precisions_,UBM.precisions_,axis=0)
            UBM = GaussianMixture(n_components = nComp, covariance_type=c.covType,
                                  weights_init= wIni,
                                  means_init=mIni,
                                  precisions_init=pIni,
                                  reg_covar=1e-6, max_iter=200, n_init=1,
                                  tol=1e-4, verbose = 2).fit(regData)
        with open(c.BNF_GMM_UBM_FILE_NAME, 'wb') as f:
            pickle.dump(UBM,f)
    
    