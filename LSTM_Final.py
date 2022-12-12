# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 20:06:29 2022

@author: ali
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense



def _run_lstm_model(X_train, X_test, y_train, y_test):
    
    with tf.device('/GPU:1'):    
    # export model if I would like 
        save_model = tf.keras.callbacks.ModelCheckpoint(monitor = "val_accuracy",
                                                              save_best_only = True,
                                                              filepath = "./LSTM_model"
                                                                )
        Model_LSTM = Sequential()
        Model_LSTM.add(Bidirectional(LSTM(units=64,
                                          recurrent_dropout=0.1,
                                          dropout=0.1,
                                          return_sequences=True),
                                         input_shape = (
                                             X_train.shape[1],
                                             X_train.shape[2]
                                            )))
            
        Model_LSTM.add(Bidirectional(LSTM(units=32,
                                          recurrent_dropout=0.1,
                                          dropout=0.1,
                                              return_sequences=False)))
    
        Model_LSTM.add(Dense(2, activation='sigmoid',))
        
        Model_LSTM.compile(optimizer="Adam",
                       loss="categorical_crossentropy",
                       metrics= ["accuracy"]
                      )
    
        #stop training if model gets worse
        Early_Stop = tf.keras.callbacks.EarlyStopping(monitor="loss", 
                                                         patience= 5,    
                                                         mode="min"
                                                         )
    
        
        history = Model_LSTM.fit(   X_train, 
                                    y_train, 
                                    epochs=100,
                                    batch_size=16,
                                    validation_data=(X_test, y_test), 
                                    callbacks= [Early_Stop, save_model])
    
    return(history)
                