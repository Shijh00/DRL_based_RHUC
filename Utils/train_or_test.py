# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 19:11:43 2021

@author: Jinshi
"""

from Utils.datasets import wind_test,wind_test_f,wind_test_r,wind_train,wind_train_f,wind_train_r,load_train,load_train_scene,load_test,load_test_scene

def train_or_test(name):
    if name == 'train':
        wind = wind_train
        wind_f = wind_train_f
        wind_r = wind_train_r
        load = load_train
        load_scene = load_train_scene
    elif name == 'test':
        wind = wind_test
        wind_f = wind_test_f
        wind_r = wind_test_r
        load = load_test
        load_scene = load_test_scene
    return wind,wind_f,wind_r,load,load_scene

#wind,wind_f,wind_r,load,load_scene = train_or_test('test')