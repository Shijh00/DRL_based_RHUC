# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 21:12:17 2021

@author: Jinshi
"""

import numpy as np
from joblib import load


def scene_decide(w_s,w_r):
    label = clf1.predict(w_s)
    wind_f = np.transpose(w_s)
    scene_1 = np.zeros((6,50))
    scene_2 = np.zeros((3,50))
    
    for i in range(6):
        for j in range(50):
            scene_1[i][j] = scene_all_100[label[0]][i][j]
    for i in range(3):
        for j in range(50):
            scene_2[i][j] = scene_r_100[label[0]][i][j]
    distance = np.zeros(50)
    for i in range(50):
        distance[i] = (2*np.linalg.norm(scene_2[:,i:i+1]-np.transpose(w_r))+np.linalg.norm(wind_f - scene_1[:,i:i+1]))/scene_p_100[label[0]][i]
    idx1 = np.argpartition(distance,9)
    scene1 = np.zeros((6,10))
    for i in range(6):
        for j in range(9):
            scene1[i][j] = scene_1[i][idx1[j]]
    scene1[:,9:10] = wind_f
    return scene1
scene_all_100 = np.load('Data//scene_all.npy')
scene_p_100 = np.load('Data//scene_p.npy')
scene_r_100 = np.load('Data//scene_r.npy')
clf1 = load('Data//kmeans-100.pkl') 
   
