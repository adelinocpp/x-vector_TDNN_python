# Third Party
import librosa
import numpy as np
import random
import math
from utils import vad
# ---- POR ADELINO ------------------------------------------------------------
class SpheringSVD:
    def __init__(self):
        self.epsilon = 1e-5
        self.Mean = 0;
        self.Z = 1;
        
    def fit(self,X):
        self.Mean = np.mean(X.T, axis=0);
        Xc = X.T - self.Mean
        Xc = Xc.T
        sigma = np.cov(Xc, rowvar=True) # [M x M]
        U,S,V = np.linalg.svd(sigma)
        # self.Z = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + self.epsilon)), U.T)) 
        self.Z = np.dot(V, np.diag(1.0/np.sqrt(S + self.epsilon)))
        
    def transform(self, X):
        Xc = (X.T - self.Mean).T
        return np.dot(self.Z, Xc)

# -----------------------------------------------------------------------------    
def mfcc(filepath, win_length=0.025, hop_length=0.01, n_mels=13):
    audio_data, fs = librosa.load(filepath, sr=None)
    n_win_length = math.ceil(fs*win_length)
    n_hop_length = math.ceil(fs*hop_length)
    n_FFT = 2 ** math.ceil(math.log2(n_win_length))
    vad_sohn = vad.VAD(audio_data, fs, nFFT=n_FFT, win_length=win_length, hop_length=hop_length, theshold=0.7)
    mfcc = librosa.feature.melspectrogram(audio_data, sr=fs, S=None, n_fft=n_FFT, n_mels=n_mels, \
                        win_length=n_win_length, hop_length=n_hop_length, window='hamming', \
                        center=False, htk=False, norm=None)
    delta1 = librosa.feature.delta(mfcc, width=3, axis=0, order=1)
    delta2 = librosa.feature.delta(mfcc, width=3, axis=0, order=2)
    mfcc = np.concatenate((mfcc,delta1, delta2), axis=0)
    if not (mfcc.shape[1] == vad_sohn.shape[0]):
        if (mfcc.shape[1] > vad_sohn.shape[0]):
            mfcc = mfcc[:,0:vad_sohn.shape[0]]
        else:
            vad_sohn = vad_sohn[0:mfcc.shape[1],:]
    return mfcc, vad_sohn
# ============================================================================  
def load_wav(audio_filepath, sr, min_dur_sec=4):
    audio_data,fs  = librosa.load(audio_filepath,sr=8000)
    len_file = len(audio_data)
    
    if len_file <int(min_dur_sec*sr):
        dummy=np.zeros((1,int(min_dur_sec*sr)-len_file))
        extened_wav = np.concatenate((audio_data,dummy[0]))
    else:
        
        extened_wav = audio_data
    return extened_wav
# -----------------------------------------------------------------------------    
def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=512):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T
# -----------------------------------------------------------------------------    
def load_data(filepath,sr=8000, min_dur_sec=4,win_length=200,hop_length=80, n_mels=13, spec_len=200,mode='train'):
    audio_data = load_wav(filepath, sr=sr,min_dur_sec=min_dur_sec)
    #linear_spect = lin_spectogram_from_wav(audio_data, hop_length, win_length, n_mels)
    linear_spect = lin_spectogram_from_wav(audio_data, hop_length, win_length, n_fft=512)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    # print("mag:",mag.shape)
    mag_T = mag.T
    
    if mode=='train':
        randtime = np.random.randint(0, mag_T.shape[1]-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        spec_mag = mag_T
    
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)
# -----------------------------------------------------------------------------        
def speech_collate(batch):
    targets = []
    specs = []
    for sample in batch:
        specs.append(sample['features'])
        targets.append((sample['labels']))
    return specs, targets
# -----------------------------------------------------------------------------    