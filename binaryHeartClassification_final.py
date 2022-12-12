    # -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:41:05 2022
@author: ali
"""
import os
import pandas as pd
from preprocess_audio import _export_features, _scale_and_encode
from LSTM_Final import _run_lstm_model

class _processingData():

    def __init__(self):
        self.data = self._join_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self._process_samples()
        self.history = self._eval_model()
    
        
    def _get_labels(self):
        path = "C:\\Users\\ali\\Desktop\\ECE Project\\ytrain"
        labels = []
        for filename in os.listdir(path):
            labels.append(os.path.join(path, filename))
        labels = pd.concat(map(pd.read_csv, labels))
        labels.columns = ['wave', 'label']
        return labels


    def _get_audio(self):
        path = "C:\\Users\\ali\\Desktop\\ECE Project\\xtrain"
        wave = []
        for filename in os.listdir(path):
            wave.append(os.path.join(path, filename))
        wave = pd.DataFrame(wave, columns=['filepath'])
        wave['wave'] = wave['filepath'].str.split("xtrain").str[1]
        wave['wave'] = wave['wave'].str[1:6]
        return wave 
        
    
    def _join_data(self):
        wave  = self._get_audio()
        labels   = self._get_labels()
        final_dataset = wave.merge(labels, on='wave', how='inner')
        return final_dataset

      
    def _process_samples(self):     
        data = self._join_data()
        xdata = []
        ydata = []
    
        for path, category_wav in zip(data.filepath, data.label):
            wav_features = _export_features(path)
    
            for indexing in wav_features:
                xdata.append(indexing)
                ydata.append(category_wav)
    
        Final_Data = pd.DataFrame(xdata)    
        Final_Data["label"] = ydata
        X_train, X_test, y_train, y_test = _scale_and_encode(Final_Data)
        
        return X_train, X_test, y_train, y_test

    def _eval_model(self):
        X_train, X_test, y_train, y_test = self._process_samples()
        history = _run_lstm_model(X_train, X_test, y_train, y_test)
        return history