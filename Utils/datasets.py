# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 18:57:33 2021

@author: Jinshi
"""

import numpy as np
import pandas as pd

history_data_Tall = 210432 - 366*24*4
T0 = 5
history_data_hours = 43824 #+366*24
train_data_Tall = 35136
test_data_Tall = 5664
train_data_hours = 8784
test_data_hours = 1416
scene_length = 6
def get_wind_history():
    f = open('Data//wind_data.csv')
    df = pd.read_csv(f)
    data0 = df.iloc[0:0+history_data_Tall:,3:T0+1].values        # the obtained data from csv file
    wind_history = np.zeros((history_data_hours,3))
    k = 0
    for i in range(history_data_Tall):
        if i % 4 == 0:
            for j in range (3):
                wind_history[k][j] =  data0[i][j]/10000
            k+=1
    return wind_history
def get_history_scene():
    wind_ = np.zeros((history_data_hours,2))
    k = 0
    for i in range(history_data_hours):
        wind_[k][0] = wind_history[i][0]/wind_history[i][2]
        wind_[k][1] = wind_history[i][1]/wind_history[i][2]
        if wind_[k][0] > 1:
            wind_[k][0] = 1
        if wind_[k][1] > 1:
            wind_[k][1] = 1
        k+=1
    history_forecast_scene = np.zeros((history_data_hours -6, 6))
    history_real_scene = np.zeros((history_data_hours -6, 6))
    for i in range (history_data_hours - 6):
        for j in range(6):
            history_forecast_scene[i][j] = wind_[i + j][0]
            history_real_scene [i][j] = wind_[i+j][1]
    return history_forecast_scene,history_real_scene  
def get_test_wind_data():
    f = open('Data//wind_data_test.csv')
    df = pd.read_csv(f)
    data0 = df.iloc[0:0+test_data_Tall:,3:T0+1].values        # the obtained data from csv file
    wind_test = np.zeros((test_data_hours,3))
    k = 0
    for i in range(test_data_Tall):
        if i % 4 == 0:
            for j in range (3):
                wind_test[k][j] = data0[i][j]/10000
            k+=1
    return wind_test
def test_scene():
    wind_r = np.zeros((test_data_hours,2))
    for i in range(test_data_hours):
        wind_r[i][0] = wind_test[i][0]/wind_test[i][2]
        wind_r[i][1] = wind_test[i][1]/wind_test[i][2]
    wind_test_f = np.zeros((test_data_hours-scene_length,scene_length))
    wind_test_r = np.zeros((test_data_hours-scene_length,scene_length))
    for i in range(test_data_hours-scene_length):
        for j in range (scene_length):
            wind_test_f[i][j] = wind_r[i+j][0]
            wind_test_r[i][j] = wind_r[i+j][1]
    return wind_test_f,wind_test_r
def get_wind_train_data():
    f = open('Data//wind_train_data.csv')
    df = pd.read_csv(f)
    data0 = df.iloc[0:0+train_data_Tall:,3:T0+1].values        # the obtained data from csv file
    wind_train = np.zeros((train_data_hours,3))
    k = 0
    for i in range(train_data_Tall):
        if i % 4 == 0:
            for j in range (3):
                wind_train[k][j] = data0[i][j]/10000
            k+=1
    return wind_train
def train_scene():
    wind_r = np.zeros((train_data_hours,2))
    for i in range(train_data_hours ):
        wind_r[i][0] = wind_train[i][0]/wind_train[i][2]
        wind_r[i][1] = wind_train[i][1]/wind_train[i][2]
    wind_train_f = np.zeros((train_data_hours - scene_length,scene_length))
    wind_train_r = np.zeros((train_data_hours - scene_length,scene_length))
    for i in range(train_data_hours - scene_length):
        for j in range (scene_length):
            wind_train_f[i][j] = wind_r[i+j][0]
            wind_train_r[i][j] = wind_r[i+j][1]
    return wind_train_f,wind_train_r
def get_load_train_data():
    f = open('Data//load_train.csv')
    df = pd.read_csv(f)
    data0 = df.iloc[0:train_data_Tall:,2:4].values        # the obtained data from csv file
    load_train = np.zeros((train_data_hours,2))
    k = 0 
    for i in range(train_data_Tall):
        if i % 4 == 0:
            load_train[k][0] = data0[i][0]/10000
            load_train[k][1] = data0[i][1]/10000
            k+=1
    return load_train
def load_train_scene():
    load_train_scene= np.zeros((train_data_hours - scene_length,scene_length))
    for i in range(train_data_hours - scene_length):
        for j in range(scene_length):
            load_train_scene[i][j] = load_train[i+j][1]
    return load_train_scene
def get_load_test_data():
    f = open('Data//load_test.csv')
    df = pd.read_csv(f)
    data0 = df.iloc[0:test_data_Tall:,2:4].values        # the obtained data from csv file
    test_hours = 1416
    load_test = np.zeros((test_hours,2))
    k = 0 
    for i in range(test_data_Tall):
        if i % 4 == 0:
            load_test[k][0] = data0[i][0]/10000 #real
            load_test[k][1] = data0[i][1]/10000#forecast
            k+=1
    return load_test
def load_test_scene():
    load_test_scene= np.zeros((train_data_hours - scene_length,scene_length))
    for i in range(1416 - scene_length):
        for j in range(scene_length):
            load_test_scene[i][j] = load_test[i+j][1]
    return load_test_scene

    
wind_history = get_wind_history()
history_forecast_scene,history_real_scene = get_history_scene()
wind_test = get_test_wind_data()
wind_test_f,wind_test_r = test_scene()
wind_train = get_wind_train_data()
wind_train_f,wind_train_r = train_scene()

load_train = get_load_train_data()
load_test = get_load_test_data()
load_train_scene = load_train_scene()
load_test_scene = load_test_scene()
