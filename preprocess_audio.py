# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 18:38:28 2022

@author: ali
"""

import librosa
import numpy as np  
from sklearn.preprocessing import OneHotEncoder #, StandardScaler
from sklearn.model_selection import train_test_split
import noisereduce as nr


sample_rate = 22050
def _get_features(y, sr):
    get_data = np.array([])
    rmse = np.mean(librosa.feature.rms(y=y))
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr))
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=y,sr=sample_rate).T,axis=0)
    get_data = np.hstack((get_data,rmse, chroma_stft, spec_cent, spec_bw, rolloff, zcr, mfcc, melspectrogram))
    return get_data    
    
def add_distribution_noise(wav):
    noise_amp  = 0.005*np.random.uniform()*np.amax(wav)
    dist_noise = wav.astype('float64') + noise_amp * np.random.normal(size=wav.shape[0])
    return dist_noise


def _export_features(path):
    
    results = []
    
    y, sr = librosa.load(path, sr=sample_rate,res_type='kaiser_fast') 
    y = nr.reduce_noise(y=y, sr=sr)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0) 
    feature = np.array(mfcc)
    results.append(feature)
         
    # features_output = _get_features(y, sr)
    # result_1 = np.array(features_output)   
    
    # features_output = add_distribution_noise(features_output)
    # results_2 = np.array(features_output)    
    
    # stretched_audio  = librosa.effects.time_stretch(y, 0.6)
    # stretched_pitch  = librosa.effects.pitch_shift(stretched_audio, sr,0.3)
    # results3 = _get_features(stretched_pitch, sr)
    # results = np.vstack((result_1, results_2, results3))
 
    return results
    

def _scale_and_encode(dataframe):
    xvals = dataframe.iloc[:,:-1].values
    
    labels = dataframe["label"].values
    labels = OneHotEncoder().fit_transform(np.array(labels).reshape(-1,1)).toarray()
    X_train, X_test, y_train, y_test =       train_test_split(xvals,        \
                                             labels,                        \
                                             train_size=0.8,               \
                                             random_state=30,               \
                                             shuffle=True)
    X_train = np.expand_dims(X_train,axis=2)
    X_test = np.expand_dims(X_test,axis=2)
    return X_train, X_test, y_train, y_test



