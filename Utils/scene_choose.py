# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 20:19:24 2021

@author: Jinshi
"""

import numpy as np
import Utils.scene_choose_0 as sg



scene_all = np.load('Data//scene_100_all.npy')#训练数据对应场景集
def scene_generation(stage,wind_f,wind_r):
    scene = sg.scene_decide(wind_f[stage+1:stage+2,:],wind_r[stage-2:stage-1,0:3])
    return scene
def scene_choose(stage):
    scene = np.zeros((6,10))
    scene =np.squeeze(scene_all[stage+1:stage+2,:,:])
    return scene

