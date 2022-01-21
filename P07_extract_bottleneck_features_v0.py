"""
Created on Sat Aug 28 10:20:11 2021

@author: adelino
"""

import torch
# import torch.nn.functional as F
# from torch.autograd import Variable

import pandas as pd
import math
import os
import configure as c
from DB_wav_reader import read_feats_structure, find_feats, find_wavs, get_num_class
from SR_Dataset import read_MFB, ToTensorTestInput, load_model, get_min_loss_model
# from model.model import background_resnet
# from P01_Train_DNN_v0 import get_min_loss_model
import numpy as np
import pickle
from utils import vad, welford
import librosa
from python_speech_features import mfcc, delta

# ----------------------------------------------------------------------------
def normalize_frames(m,ubm_mean,ubm_std):
    return (m - ubm_mean) / (ubm_std + 2e-12)
# ----------------------------------------------------------------------------
def compute_bottleneck_features(use_cuda, model, filename, embedding_size, test_frames):

    # n_win_length = math.ceil(sample_rate*win_length)
    # n_FFT = 2 ** math.ceil(math.log2(n_win_length))
    # audio, sr = librosa.load(filename, sr=sample_rate, mono=True)    
    # n_hop_length = math.ceil(sample_rate*hop_length)    
    # linear_spect = librosa.stft(audio, hop_length=n_hop_length, win_length=n_win_length, n_fft=n_FFT, window='hamming')
    # mag, _ = librosa.magphase(linear_spect)  # magnitude
    # mag = mag.T
    # mag_delta = delta(mag,2)    
    # mag_delta2 = delta(mag_delta,2)
    # mfcc_feat = np.concatenate((mag, mag_delta, mag_delta2),axis=1)
    
    # vad_sohn = vad.VAD(audio, sr, nFFT=n_FFT, win_length=win_length, \
    #                    hop_length=hop_length, theshold=0.7)
    
    mfcc_feat, label = read_MFB(filename)
    tot_segments = mfcc_feat.shape[0] - test_frames # total number of segments with 'test_frames' 
    # tot_segments = mfcc_feat.shape[0]
    output = torch.zeros((tot_segments,embedding_size))
    
    with torch.no_grad():
        for i in range(0,tot_segments):
            # temp_input = mfcc_feat[i,:]
            # temp_input.shape = (temp_input.shape[0],1)
            temp_input = mfcc_feat[i:i+test_frames]
            # print("temp_input: input shape: {:} totseg: {:} len: {:} shape: {:}".format(input.shape,tot_segments, len(temp_input.shape),temp_input.shape))
            TT = ToTensorTestInput()
            temp_input = TT(temp_input) # size:(1, 1, n_dims, n_frames)
    
            if use_cuda:
                temp_input = temp_input.cuda()
            activation,_ = model(temp_input)            
            output[i,:] = activation.squeeze()
    
    # if not (output.shape[0] == vad_sohn.shape[0]):
    #     if (output.shape[0] > vad_sohn.shape[0]):
    #         output = output[0:vad_sohn.shape[0],:]
    #     else:
    #         vad_sohn = vad_sohn[0:output.shape[0],:]
    
    # vadIDX = (vad_sohn == 1).nonzero()[0]
    # output = output[vadIDX,:] 
    output = l2_norm(output, c.ALPHA_LNORM)
    output = output.numpy()
    output_delta = delta(output,2)    
    output_delta2 = delta(output_delta,2)
    output_feat = np.concatenate((output, output_delta, output_delta2),axis=1)
    return output_feat
# ----------------------------------------------------------------------------
def l2_norm(input, alpha):
    input_size = input.size()  # size:(n_frames, dim)
    buffer = torch.pow(input, 2)  # 2 denotes a squared operation. size:(n_frames, dim)
    normp = torch.sum(buffer, 1).add_(1e-10)  # size:(n_frames)
    norm = torch.sqrt(normp)  # size:(n_frames)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
    output = output * alpha
    return output
# ------------------------------------------------------------------------------

Compute_BNF_Train_UBM_Mean_Std = True
Normalize_Train_BNF = True
Compute_Norm_Test_BNF = True
use_cuda = False
log_dir = c.SAVE_MODELS_DIR
embedding_size = 512
# --- dimensao 129 ---------------------------------------------------------
featureDim = 3*129

cp_num,__ = get_min_loss_model(log_dir) # Which checkpoint to use?
n_classes = get_num_class(c.TRAIN_FEAT_DIR)
# n_classes = 104
model = load_model(use_cuda, log_dir, cp_num, featureDim, n_classes)
test_frames = c.NUM_WIN_SIZE

if (Compute_BNF_Train_UBM_Mean_Std):
    w_ubm = welford.Welford()
    file_list = find_feats(c.TRAIN_FEAT_DIR)
    splitDir = c.TRAIN_BNF_DIR.split('/')
    for idx in range(1,len(splitDir)+1):
        curDir = '/'.join(splitDir[:idx])
        if (not os.path.exists(curDir)):
            os.mkdir(curDir)
    
    for idx, filename in enumerate(file_list):
        print('Iniciado arquivo   {:4} de {:4}'.format(idx, len(file_list)));
        feat_and_label = {}
        features = compute_bottleneck_features(use_cuda, model, filename, embedding_size, test_frames)
        
        # if (features.shape[0] < c.NUM_WIN_SIZE):
        #     print('Finalizado arquivo {:4} de {:4} - Tamanho inadequado'.format(idx, len(file_list)-1));
        #     continue
        
        filenameParts = filename.replace('\\', '/')
        filenameFolder = filenameParts.split('/')[-2]
        filenameBase = filenameParts.split('/')[-1].split('.')[0]
        filenameSave = c.TRAIN_BNF_DIR + '/' + filenameFolder + '/' + filenameBase + '.p'
        w_ubm.update(features)
        feat_and_label['feat'] = features
        feat_and_label['label'] = filenameFolder
        if (not os.path.exists(c.TRAIN_BNF_DIR + '/' + filenameFolder)):
            os.mkdir(c.TRAIN_BNF_DIR + '/' + filenameFolder)
        with open(filenameSave, 'wb') as f:
            pickle.dump(feat_and_label,f)
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));
        # if (idx == 2):
        #     break
    ubm_data = {}
    ubm_data["mean"]    = w_ubm.mean
    ubm_data["std"]     = w_ubm.std

    with open(c.BNF_UBM_FILE_NAME, 'wb') as f:
        pickle.dump(ubm_data,f)
else:
    with open(c.BNF_UBM_FILE_NAME, 'rb') as f:
        ubm_data = pickle.load(f)

print('BNF Train calculados...')

if (Normalize_Train_BNF):
    ubmDim = ubm_data["mean"].shape[0]
    print('Normalização das BNF de treinamento...')
    file_list = find_feats(c.TRAIN_BNF_DIR)
    print('Arquivos para normalizar {}'.format(len(file_list)))
    for idx, filename in enumerate(file_list):
        print('Iniciado arquivo   {:4} de {:4}'.format(idx, len(file_list)-1));
        with open(filename, 'rb') as f:
            feat_and_label = pickle.load(f)
            
        Fndim = feat_and_label['feat'].shape[1]
        
        if (Fndim == ubmDim):
            feat_and_label['feat'] = normalize_frames(feat_and_label['feat'],\
                                        ubm_data["mean"],ubm_data["std"])
            with open(filename, 'wb') as f:
                pickle.dump(feat_and_label,f)
        # else:
        #     os.remove(filename)
        
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1));

print('BNF Train normalizados...')

if (Compute_Norm_Test_BNF):
    splitDir = c.TEST_BNF_DIR.split('/')
    for idx in range(1,len(splitDir)+1):
        curDir = '/'.join(splitDir[:idx])
        if (not os.path.exists(curDir)):
            os.mkdir(curDir)
            
    ubmDim = ubm_data["mean"].shape[0]
    print('Inicio do calculo das BNFs de teste:')
    file_list = find_feats(c.TEST_FEAT_DIR)
    print('Arquivos de teste {}'.format(len(file_list)))
    for idx, filename in enumerate(file_list):
        print('Iniciado arquivo   {:4} de {:4}'.format(idx, len(file_list)));
        feat_and_label = {}
        features =  compute_bottleneck_features(use_cuda, model, filename, embedding_size, test_frames)
        # features = compute_features(filename,c.SAMPLE_RATE,c.USE_SCALE) 
        if (features.shape[0] < c.NUM_WIN_SIZE):
            print('Finalizado arquivo {:4} de {:4} - Tamanho inadequado'.format(idx, len(file_list)-1));
            continue
        filenameParts = filename.replace('\\', '/')
        filenameFolder = filenameParts.split('/')[-2]
        filenameBase = filenameParts.split('/')[-1].split('.')[0]
        filenameSave = c.TEST_BNF_DIR + '/' + filenameFolder + '/' + filenameBase + '.p'
        
        Fndim = features.shape[1]
        
        if (Fndim == ubmDim):
            feat_and_label['feat'] = normalize_frames(features,\
                                        ubm_data["mean"],ubm_data["std"])
            feat_and_label['label'] = filenameFolder
            if (not os.path.exists(c.TEST_BNF_DIR + '/' + filenameFolder)):
                os.mkdir(c.TEST_BNF_DIR + '/' + filenameFolder)
                with open(filenameSave, 'wb') as f:
                    pickle.dump(feat_and_label,f)
        
        print('Finalizado arquivo {:4} de {:4}'.format(idx, len(file_list)-1))
        
print('BNF Test calculados...')