# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:25:20 2021

@author: Administrator
"""

import tensorflow as tf
import numpy as np

MAX_EPISODES = 12000 #  the max number of learning 
MAX_EP_STEPS = 24 # the number of dispatch
LR_A = 0.001  # learning rate for actor
LR_C = 0.0001   # learning rate for critic
GAMMA = 0.99    # reward discount
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

# the definition of DDPG
class TD3(object):
    def __init__(self, a_dim, s1_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s1_dim*2 + a_dim + 1), dtype=np.float32)#经验回放池
        self.memory2 = np.zeros((MEMORY_CAPACITY,6,22), dtype=np.float32)
        self.pointer = 0 #内部指针，用于记录放入与外界交互次数
        self.sess = tf.Session()
        self.a_dim, self.s_dim, self.a_bound = a_dim, s1_dim, a_bound,  #a_dim is the dimension of action and so on
        self.S = tf.placeholder(tf.float32, [None, s1_dim], 's') #用于存放状态s
        self.S2 = tf.placeholder(tf.float32, [None,6,11], 's2')
        
        self.S_ = tf.placeholder(tf.float32, [None, s1_dim], 's_') #用于存放下一时刻的状态s_
        self.S2_ = tf.placeholder(tf.float32, [None,6,11], 's2_')
       
        self.R = tf.placeholder(tf.float32, [None, 1], 'r') #用于存放动作a的奖励r
        #建立actor网络 a为当前网络，a_为目标网络
        with tf.variable_scope('Actor',reuse = tf.AUTO_REUSE):
            self.a = self._build_a(self.S, self.S2,scope='eval', trainable=True)
            a_ = self._build_a(self.S_,self.S2_, scope='target', trainable=False)
            a_ = a_ + np.clip(np.random.normal(0,0.05,size=(a_dim,)), -1, 1)
            a_ = tf.clip_by_value(a_, -1,1)
        #建立critic网络，q为当前网络，q_为目标网络
        with tf.variable_scope('Critic',reuse = tf.AUTO_REUSE):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q = self._build_c(self.S,self.S2, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, self.S2_,a_ , scope='target', trainable=False)
        with tf.variable_scope('Critic2',reuse = tf.AUTO_REUSE):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q_2 = self._build_c(self.S, self.S2,self.a, scope='eval', trainable=True)
            q_2_ = self._build_c(self.S_, self.S2_,a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')
        self.ce2_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic2/eval')
        self.ct2_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic2/target')
        self.copy =  [tf.assign(t, e)
                             for t, e in zip(self.at_params + self.ct_params + self.ct2_params, self.ae_params + self.ce_params +  self.ce2_params)]
        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params + self.ct2_params, self.ae_params + self.ce_params +  self.ce2_params)]
        q_t = tf.where(q_<q_2_,q_,q_2_) 
        self.q_target = self.R + GAMMA*q_t
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        self.td_error = tf.losses.mean_squared_error(self.q_target, self.q)
        self.td_error2 = tf.losses.mean_squared_error(self.q_target, self.q_2)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.td_error, var_list=self.ce_params)
        self.c2train = tf.train.AdamOptimizer(LR_C).minimize(self.td_error2, var_list=self.ce2_params)
       # self.a_loss = -tf.losses.mean_squared_error(self.q , self.q-self.q)
        self.a_loss =  -tf.reduce_mean(self.q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=self.ae_params)
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s,s2):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :],self.S2: s2[np.newaxis, :] })[0]
    def init_copy(self):
        self.sess.run(self.copy)
    def learn_a(self):
        # soft target replacement
        indices = np.random.choice(min(self.pointer,MEMORY_CAPACITY), size=BATCH_SIZE,replace = False)
        bt= self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        bs2 = self.memory2[indices, :,0:11]
        self.sess.run(self.atrain, {self.S: bs,self.S2 :bs2})
        self.sess.run(self.soft_replace)
    def learn_c(self):
        # soft target replacement
        indices = np.random.choice(min(self.pointer,MEMORY_CAPACITY), size=BATCH_SIZE,replace = False)
        bt= self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        bs2 = self.memory2[indices, :,0:11]
        bs2_ =self.memory2[indices, :,11:22]
        #print(self.sess.run(self.a_loss,{self.S:bs,self.a:ba}))
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_,self.S2:bs2,self.S2_:bs2_})
        self.sess.run(self.c2train, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_,self.S2:bs2,self.S2_:bs2_})
        
        if (self.pointer+1) % 600== 0:
            print(self.sess.run(self.td_error,{self.S: bs, self.a: ba, self.R: br, self.S_: bs_,self.S2:bs2,self.S2_:bs2_}),
                  self.sess.run(self.td_error2,{self.S: bs, self.a: ba, self.R: br, self.S_: bs_,self.S2:bs2,self.S2_:bs2_}),
                  )
        
            
    def store_transition(self, s,a, r, s_, s2,s2_):
        transition = np.hstack((s, a, [r] , s_))
        transition2 = np.hstack((s2, s2_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.memory2[index, :,:] = transition2
        self.pointer += 1
    
    def _build_a(self, s,s2, scope, trainable):
        with tf.variable_scope(scope):
            output, st = tf.nn.dynamic_rnn(tf.nn.rnn_cell.GRUCell(100), s2, dtype=tf.float32)
            n_l1 = 2**8
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_s2 = tf.get_variable('w1_s2', [100, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(st, w1_s2) + b1)
            net2 = tf.layers.dense(net1, 2**8, activation=tf.nn.relu, name='l2', trainable=trainable)
            net3 = tf.layers.dense(net2, 2**8, activation=tf.nn.relu, name='l3', trainable=trainable)
            #net4 = tf.layers.dense(net3, 20, activation=tf.nn.relu, name='l4', trainable=trainable)
            
            a = tf.layers.dense(net3, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')
    
    def _build_c(self, s,s2, a, scope, trainable):
        with tf.variable_scope(scope):
            output, st = tf.nn.dynamic_rnn(tf.nn.rnn_cell.GRUCell(100), s2, dtype=tf.float32)
            n_l1 = 2**9
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_s2 = tf.get_variable('w1_s2', [100, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a)+tf.matmul(st, w1_s2) + b1)
        
            critic_layer2 = tf.layers.dense(net, 2**8, activation = tf.nn.relu, name='c_l2', trainable=trainable)
            critic_layer3 = tf.layers.dense(critic_layer2, 2**8, activation = tf.nn.relu, name='c_l3', trainable=trainable)
            return tf.layers.dense(critic_layer3, 1, trainable=trainable)  # Q(s,a)
    def _build_c_2(self, s, s2,a,scope, trainable):
        with tf.variable_scope(scope):
            output, st = tf.nn.dynamic_rnn(tf.nn.rnn_cell.GRUCell(100), s2, dtype=tf.float32)
            n_l1 = 2**9
            w1_s = tf.get_variable('c_2_w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('c_2_w1_a', [self.a_dim, n_l1], trainable=trainable)
            w1_s2 = tf.get_variable('c_2_w1_s2', [100, n_l1], trainable=trainable)
            b1 = tf.get_variable('c_2_b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a)+tf.matmul(st, w1_s2) + b1)
        
            critic_layer2 = tf.layers.dense(net, 2**8, activation = tf.nn.relu, name='c_2_l2', trainable=trainable)
            critic_layer3 = tf.layers.dense(critic_layer2, 2**8, activation = tf.nn.relu, name='c_2_l3', trainable=trainable)
            return tf.layers.dense(critic_layer3, 1, trainable=trainable)  # Q(s,a)