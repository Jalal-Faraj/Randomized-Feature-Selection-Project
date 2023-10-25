#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:52:58 2021

@author: jalalfaraj
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import pickle

# Randomly select 10 features in each step, choose best feature based on model performance, and move on to the next
# feature, for a total of n features:
def mean_relative_error(y_test, pred):
    y_test_array = np.array(y_test)
    mre_error = abs(pred.flatten() - y_test_array)/abs(y_test_array)
    mre = np.sum(mre_error)/pred.size
    return mre

def model(X1, Y):
    x_train, x_test, y_train, y_test = train_test_split(X1, Y, test_size=0.33, random_state=42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # from tensorflow.keras.callbacks import ModelCheckpoint
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(128, kernel_initializer='normal',input_dim = x_train.shape[1], activation='relu'))

    # The Hidden Layers :
    NN_model.add(Dense(64, kernel_initializer='normal',activation='linear'))
    NN_model.add(Dense(64, kernel_initializer='normal',activation='linear'))
    NN_model.add(Dense(64, kernel_initializer='normal',activation='linear'))
    NN_model.add(Dense(64, kernel_initializer='normal',activation='linear'))


    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    # Compile the network :
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae', 'mse'])
    #NN_model.summary()
    NN_model.fit(x_train, y_train, epochs=10)
    
    pred = NN_model.predict(x_test)
    
    res = mean_relative_error(y_test, pred)
    return res
    
def best_feat(df, num_feats , best_n_feats, X):
    # determins best local feature, and references previous features
    res = []
    Y = pd.read_csv('out-0812-pdi.csv')
    for z in range(df.shape[1]):
        X1 = df[[df.columns[z]]]
        if num_feats != 0:
            prev_feats = X[best_n_feats]
            X_com = pd.concat([X1, prev_feats], axis = 1)
            acc = model(X_com,Y)
        else: 
            acc = model(X1,Y)
            X_com = []
        res.append(acc)
        best = min(res)
        best_ft = res.index(best)
    return best_ft

#def best_feats(df):
    # determine best new feature by 
    
def for_feat_sel(num_of_feats):
    df_new_ft = []
    X = pd.read_csv('All Features.csv')
    X = X.drop(['Ref','Buffer'],axis = 1)
    X_fs = X
    while len(df_new_ft) < num_of_feats:
#     for i in range(100):
        # Randomly select 10 features to compare their performance
        df = X.sample(n=25, replace = 'True', axis = 'columns', random_state = 15)
        num_feats = len(df_new_ft)
        ind_bf = best_feat(df, num_feats, df_new_ft, X_fs)
        
        # Only add feature if it is not present in the best features list
        if df.columns[ind_bf] not in df_new_ft:
            df_new_ft.append(df.columns[ind_bf])
            print('Adding feature: ',df.columns[ind_bf])
  
        Xnew = X.drop([df.columns[ind_bf]],axis = 1)
        X = Xnew
            
    return df_new_ft

if __name__ == "__main__":          
    feats_50 = for_feat_sel(50)
        
    with open('feature_sel_feats.pkl', 'wb') as f:
        pickle.dump(feats_50, f)

