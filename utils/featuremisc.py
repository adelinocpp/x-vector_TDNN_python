# -*- coding: utf-8 -*-
# -------------------------------
import os
import librosa
import numpy as np
from utils import utils
# from utils.utils import speech_collate
import pickle
import sys
import random
# -------------------------------
# import torch
# from torch import optim
# import torch.nn as nn
# from torch.utils.data import DataLoader
# -----------------------------------------------------------------------------
# from sklearn.metrics import accuracy_score
# -----------------------------------------------------------------------------
# from models.x_vector import X_vector

# -----------------------------------------------------------------------------
def feature_extraction(fileName):
    if (type(fileName) is str):
        return utils.mfcc(fileName)
    if (type(fileName) is list):
        listofMFCC = []
        listofVAD = []
        nFiles = len(fileName)
        idxFile = 0
        for file in fileName:
            mfcc, vad = utils.mfcc(file)
            listofMFCC.append(mfcc)
            listofVAD.append(vad)
            print("\tfinalizado",idxFile,"de",nFiles,". File : ",file)
            idxFile += 1
        return listofMFCC, listofVAD
# -----------------------------------------------------------------------------
def compute_and_save_raw_features(dataFolder,fileList,RAW_MFCC_FILE,RAW_VAD_FILE):
    if not (os.path.isfile(dataFolder + RAW_MFCC_FILE) & os.path.isfile(dataFolder + RAW_VAD_FILE)):
        print("Calcula...", dataFolder + RAW_MFCC_FILE)
        MFCC, VAD = feature_extraction(fileList)
        with open(dataFolder + RAW_MFCC_FILE, "wb") as fp: 
            pickle.dump(MFCC, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()
        with open(dataFolder + RAW_VAD_FILE, "wb") as fp: 
            pickle.dump(VAD, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()
# -----------------------------------------------------------------------------
def load_UBM(dataFolder, filenameUBM):
    if not os.path.isfile(dataFolder + filenameUBM):
        return False    
    with open(dataFolder + filenameUBM, "rb") as fp: 
        ubm_mfcc = pickle.load(fp)    
        fp.close()
    return ubm_mfcc
# -----------------------------------------------------------------------------
def build_and_save_UBM(dataFolder, filenameMFCC, filenameVAD, filenameUBM):
    if not ((type(filenameMFCC) is str) & (type(filenameVAD) is str)):
        return False
    if os.path.isfile(dataFolder + filenameUBM):
        return False
    existMFCC   = os.path.isfile(dataFolder + filenameMFCC)
    existVAD    = os.path.isfile(dataFolder + filenameVAD)
    if not (existMFCC & existVAD):
        return False
    with open(dataFolder + filenameMFCC, "rb") as fp: 
        MFCC_DATA = pickle.load(fp)
        fp.close()
    with open(dataFolder + filenameVAD, "rb") as fp: 
        VAD_DATA = pickle.load(fp)
        fp.close()
    if not (len(MFCC_DATA) == len(VAD_DATA)):
        return False

    ubm_mfcc = np.empty([MFCC_DATA[0].shape[0], 0])
    for i in range(0,len(MFCC_DATA)):
        mfcc = MFCC_DATA[i]
        vad = VAD_DATA[i]
        if (not (mfcc.shape[1] == vad.shape[0])):
            print("(mfcc.shape[1] == vad.shape[0]), i",i)
            return False
        vadIDX = (vad == 1).nonzero()[0]
        ubm_mfcc = np.concatenate((ubm_mfcc,mfcc[:,vadIDX]), axis=1) 
    
    with open(dataFolder + filenameUBM, "wb") as fp: 
        pickle.dump(ubm_mfcc, fp, protocol=pickle.HIGHEST_PROTOCOL)
        fp.close()

    return True
# -----------------------------------------------------------------------------    
def normalize_and_save_features(dataFolder, filenameMFCC, filenameVAD, filenameUBM, finenameNORM):
    if not ((type(filenameMFCC) is str) & (type(filenameVAD) is str) & (type(filenameUBM) is str)):
        return False
    if os.path.isfile(dataFolder + finenameNORM):
        return False
    with open(dataFolder + filenameMFCC, "rb") as fp: 
        MFCC_DATA = pickle.load(fp)
        fp.close()
    with open(dataFolder + filenameVAD, "rb") as fp: 
        VAD_DATA = pickle.load(fp)
        fp.close()
    with open(dataFolder + filenameUBM, "rb") as fp: 
        UBM_DATA = pickle.load(fp)
        fp.close()
    if not (len(MFCC_DATA) == len(VAD_DATA)):
        return False

    mean_ubm    = np.mean(UBM_DATA,axis=1, keepdims=True)
    std_ubm     = np.std(UBM_DATA,axis=1, keepdims=True)
    # print("mean_ubm",mean_ubm.shape)
    # print("std_ubm" ,std_ubm.shape)
    norm_mfcc = []
    for i in range(0,len(MFCC_DATA)):
        mfcc = MFCC_DATA[i]
        vad = VAD_DATA[i]
        # print("INI mfcc: ",mfcc.shape)
        if (not (mfcc.shape[1] == vad.shape[0])):
            print("(mfcc.shape[1] == vad.shape[0]), i",i)
            return False
        vadIDX = (vad == 1).nonzero()[0]
        norm_mfcc.append(np.divide(np.subtract(mfcc[:,vadIDX],mean_ubm[0]), std_ubm[0]+1e-9))
        # norm_mfcc.append(np.divide(np.subtract(mfcc[:,vadIDX],np.repeat(mean_ubm[0],len(vadIDX))), np.repeat(std_ubm[0],len(vadIDX))))
        # print("FIM mfcc: ",MFCC_DATA[i].shape)
        
    with open(dataFolder + finenameNORM, "wb") as fp: 
        pickle.dump(norm_mfcc, fp, protocol=pickle.HIGHEST_PROTOCOL)
        fp.close()
    return True

# -----------------------------------------------------------------------------    
def load_norm_mfcc(class_index,fileData,spec_len=1000,mode='train'):
    
    with open(fileData, "rb") as fp: 
        MFCC_DATA = pickle.load(fp)
        fp.close()
    
    mag_T = MFCC_DATA[class_index]

    if mode=='train':
        if (mag_T.shape[1] > spec_len):
            idxA = np.random.permutation(mag_T.shape[1])[0:spec_len]
            # randtime = np.random.randint(0, mag_T.shape[1]-spec_len)
            # spec_mag = mag_T[:, randtime:randtime+spec_len]
        else:
            idxA = np.random.permutation(mag_T.shape[1])
        spec_mag = mag_T[:, idxA]
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    # mu = np.mean(spec_mag, 0, keepdims=True)
    # std = np.std(spec_mag, 0, keepdims=True)
    return spec_mag