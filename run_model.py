# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:22:56 2022

@author: ali
"""
from matplotlib import pyplot as plt
from binaryHeartClassification_final import _processingData

data = _processingData()

model_history = data.history
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss']) 
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
