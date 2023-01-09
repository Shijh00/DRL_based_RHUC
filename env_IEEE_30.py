# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:39:43 2022

@author: Jinshi
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Utils.scene_choose as sc
import Utils.train_or_test as tt
import TD3_gru
from TD3_gru import MAX_EP_STEPS
import tensorflow as tf
from Utils.datasets import train_data_hours
import math
import time
MAX_EPISODES = 10000
index = [0,2,1,3,4] #机组大小排序，
wind,wind_f,wind_r,load,load_scene = tt.train_or_test('train') #只有wind load load_scene 被使用
#进行负载、风电数据的放缩处理
load = load*10/4
load_scene = load_scene*10/4

#获取系统的参数
f = open('Data//unit_6.csv')
df = pd.read_csv(f)
data0 = df.iloc[0:6,0:12].values        # the obtained data from csv file
unit_para = np.zeros((6,12))
for i in range(6):
    for j in range(12):
        unit_para[i][j] = data0[i][j]
f = open('Data//netpara_30.csv')
df = pd.read_csv(f)
data0 = df.iloc[0:41,0:7].values        # the obtained data from csv file
net_para = np.zeros((41,7))
for i in range(41):
    for j in range(7):
        net_para[i][j] = data0[i][j]
f = open('Data//load_tran.csv')
df = pd.read_csv(f)
data0 = df.iloc[0:30,0:1].values  
tran = np.zeros(30)
for i in range(30):
    tran[i] = data0[i][0]
branch_num = 41 #支路数目
gennum = 6 #机组数目
unit_num = gennum -1 #火电机组数目
PL_max = net_para[:,5:6] #支路功率上限
PL_min = net_para[:,6:7] #支路功率下限
unit_limit = unit_para[:,2:4] #机组功率上下限
cost_para = unit_para[:,4:7] #机组价格参数
lasttime = unit_para[:,8:9] #机组开关机时间限制
lasttime =lasttime/100 #数据处理方便强化学习
start_cost0 = unit_para[:,9:10] #机组开机成本
initial_time = unit_para[:,11:12] #机组初始状态
initial_time = initial_time/100
#将风电机组替换掉火电机组后，对剩余的火电机组重新排序
####
wind_location = 3#风电组替换的位置 0,1,2,3,4,5
####
wind[:,0:2] = unit_limit[wind_location][0]*wind[:,0:2]/wind[:,2:3]  #对风电进行数值转化
####
on_off_time = np.delete(lasttime,wind_location )
min_and_max = np.delete(unit_limit,wind_location ,axis=0)
cost = np.delete(cost_para,wind_location ,axis = 0)
ini_time = np.delete(initial_time,wind_location )
start_cost = np.delete(start_cost0,wind_location )

ramping = np.delete(unit_para[:,7:8],wind_location )#爬坡率

numnodes = 30 #节点个数
dianna = 1/net_para[:,3:4] #电抗
slack_bus=26 #平衡节点
Y = np.zeros((numnodes,numnodes))
for k in range(branch_num):
    i = int(net_para[k][1])-1
    j = int(net_para[k][2])-1
    Y[i][j] = -dianna[k]
    Y[j][i] = Y[i][j]
s = np.sum(Y,axis =0)
for k in range(numnodes):
    Y[k][k] = -s[k]
Y = np.delete(Y,slack_bus-1, axis = 0)
Y = np.delete(Y,slack_bus-1, axis = 1)
X = np.linalg.inv(Y)
row = np.zeros((1,numnodes-1))
X = np.insert(X,slack_bus-1,row,axis = 1)
X = np.insert(X,slack_bus-1,0,axis = 0)
G = np.zeros((branch_num,numnodes))
for k in range(branch_num):
    m = int(net_para[k][1])-1
    n = int(net_para[k][2])-1
    xk = dianna[k]
    for i in range(numnodes):
        G[k][i] = (X[m][i]-X[n][i])*xk
power_gen = unit_para[:,1:2]
cut_load = 12000
cut_wind = 8000
cost_up = 4000
cost_down = 4000

def trans_load(load):
    load_distribute = np.zeros(numnodes)
    for i in range(numnodes):
        load_distribute [i] = load*tran[i]
    return load_distribute
def get_cost(state_t): #state 5*(运行时间+出力+备用) 4*gennum代表风电  load_distribute 六个机组，第二个机组被风电机组取代
    load = state_t[4*unit_num+1]#表示负载的状态
    #"""
    load_distribute = trans_load(load)
    sum_nodeGSDF = np.zeros(branch_num)
    sum_PowerGSDF = np.zeros((branch_num,numnodes))
    num = 0
    for k in range(branch_num):
        for i in range(gennum):
            if i < wind_location :
                sum_PowerGSDF[k][i] = G[k][int(power_gen[i])-1]*state_t[i+unit_num]
            elif i == wind_location:
                sum_PowerGSDF[k][i] = G[k][int(power_gen[i])-1]*state_t[4*unit_num]
            elif i> wind_location:
                sum_PowerGSDF[k][i] = G[k][int(power_gen[i])-1]*state_t[i-1+unit_num]
        for i in range(numnodes):
            sum_nodeGSDF[k] = sum_nodeGSDF[k]+G[k][i]*load_distribute[i]
    Pf = np.sum(sum_PowerGSDF,axis = 1) - sum_nodeGSDF
    #"""
    r = 0
    for i in range(branch_num ):
        if  not(Pf[i]>= PL_min[i]-0.2 and Pf[i]<=PL_max[i]+0.2):
            num+=1
            print(i+1)
    power = state_t[4*unit_num]
    wind_power = state_t[4*unit_num]
    r_up = 0
    r_down = 0
    for i in range(unit_num):
        if state_t[i] > 0:
            if state_t[i] == 0.001:
                r+=start_cost[i]
            power+=state_t[i+unit_num]
            r+=state_t[i+unit_num]*state_t[i+unit_num]*cost[i][0] + state_t[i+unit_num]*cost[i][1]+cost[i][2]
            r_up +=state_t[i+unit_num*2]
            r_down+=state_t[i+unit_num*3]
    r_up = min(r_up,state_t[4*unit_num+1]/10)
    r_down = min(r_down,state_t[4*unit_num+1]/10)
    if power > load:
        if power - load <r_down:
            r+= cost_down*(power-load)
        elif power - load - r_down< wind_power:
            r+=cost_down*r_down
            r+=cut_wind*(power - load - r_down)
        else:
            r+=cost_down*r_down
            r+=cut_wind*wind_power
    else:
        if load - power < r_up:
            r+=cost_up*(load-power)
        else:
            r+=cost_up*r_up
            r+=cut_load*(load-power-r_up)
    r = r/10000
    return r,num

def get_reward(state_t): #state 5*(运行时间+出力+备用) 4*gennum代表风电  load_distribute 六个机组，第二个机组被风电机组取代
    load = state_t[4*unit_num+1]#表示负载的状态
    #"""
    load_distribute = trans_load(load)
    sum_nodeGSDF = np.zeros(branch_num)
    sum_PowerGSDF = np.zeros((branch_num,numnodes))
    num = 0
    for k in range(branch_num):
        for i in range(gennum):
            if i < wind_location :
                sum_PowerGSDF[k][i] = G[k][int(power_gen[i])-1]*state_t[i+unit_num]
            elif i == wind_location:
                sum_PowerGSDF[k][i] = G[k][int(power_gen[i])-1]*state_t[4*unit_num]
            elif i> wind_location:
                sum_PowerGSDF[k][i] = G[k][int(power_gen[i])-1]*state_t[i-1+unit_num]
        for i in range(numnodes):
            sum_nodeGSDF[k] = sum_nodeGSDF[k]+G[k][i]*load_distribute[i]
    Pf = np.sum(sum_PowerGSDF,axis = 1) - sum_nodeGSDF
    #"""
    r = 0
    for i in range(branch_num):
        if  not(Pf[i]>= PL_min[i] and Pf[i]<=PL_max[i]):
            r+= 1000
            num+=1
    power = state_t[4*unit_num]
    wind_power = state_t[4*unit_num]
    r_up = 0
    r_down = 0
    for i in range(unit_num):
        if state_t[i] > 0:
            if state_t[i] == 0.001:
                r+=start_cost[i]
            power+=state_t[i+unit_num]
            r+=state_t[i+unit_num]*state_t[i+unit_num]*cost[i][0] + state_t[i+unit_num]*cost[i][1]+cost[i][2]
            r_up +=state_t[i+unit_num*2]
            r_down+=state_t[i+unit_num*3]
    r_up = min(r_up,state_t[4*unit_num+1]/10)
    r_down = min(r_down,state_t[4*unit_num+1]/10)
    if power > load:
        if power - load <r_down:
            r+= cost_down*(power-load)
        elif power - load - r_down< wind_power:
            r+=cost_down*r_down
            r+=cut_wind*(power - load - r_down)
        else:
            r+=cost_down*r_down
            r+=cut_wind*wind_power
    else:
        if load - power < r_up:
            r+=cost_up*(load-power)
        else:
            r+=cost_up*r_up
            r+=cut_load*(load-power-r_up)
    r = -1*(r/10000)
    return r,num
def update_state(state,action):
    state_ = np.zeros((state.shape[0]))
    for i in range(state.shape[0]):
        state_[i] = state[i] #复制state,以便于计算state t+1
    for i in range(unit_num):
        if action[i] < -0.9 : 
            if state_[i] < 0:
                state_[i] -= 0.01
            elif state_[i] >= 0 and state[i] < on_off_time[i]:
                state_[i] += 0.01
                state_[i + unit_num*2] += action[i+unit_num]*ramping[i]
                state_[i + unit_num*3] += action[i+2*unit_num]*ramping[i]
                state_[i + unit_num] -= ramping[i]
            elif state_[i] >= on_off_time[i]:
                state_[i] = -0.001
                state_[i+unit_num] = 0
                state_[i+2*unit_num] = 0
                state_[i+3*unit_num] = 0
        elif action[i]> 0.9:
            if state_[i] <= -1*on_off_time[i] :
                state_[i] = 0.001
                state_[i + unit_num] = min_and_max[i][1]
                state_[i + unit_num*2] = action[i+unit_num]*ramping[i]
                state_[i + unit_num*3] = 0
            elif state_[i] > -1*on_off_time[i] and state_[i]<0:
                state_[i] -= 0.01
            elif state_[i] >= 0:
                state_[i] += 0.01
                state_[i + unit_num*2] += action[i+unit_num]*ramping[i]
                state_[i + unit_num*3] += action[i+2*unit_num]*ramping[i]
                state_[i + unit_num] += ramping[i]
        else:
            if state_[i] < 0 :
                state_[i] -= 0.01
            elif state_[i] >= 0:
                state_[i] += 0.01
                state_[i + unit_num*2] += action[i+unit_num]*ramping[i]
                state_[i + unit_num*3] += action[i+2*unit_num]*ramping[i]
                state_[i + unit_num] += action[i]/0.9 *ramping[i]
    state_ = clip_s(state_)   
    return state_
def clip_s(state):
    state0 = np.zeros((state.shape[0]))
    for i in range(state.shape[0]):
        state0[i] = state[i]
    for i in range(unit_num):
        if state0[i] > (0.24 + on_off_time[i]):
            state0[i] = on_off_time[i]
        if -1*state0[i] > (0.24 + on_off_time[i]):
            state0[i] = -1*on_off_time[i]
    for i in range(unit_num):
        if state0[i] > 0:
            state0[i + unit_num] = min(state0[i + unit_num],min_and_max[i][0])
            state0[i + unit_num] = max(state0[i + unit_num],min_and_max[i][1])
            state0[i + 2*unit_num] = max(state0[i + 2*unit_num],0)
            state0[i + 3*unit_num] = max(state0[i + 3*unit_num],0)
            state0[i + 2*unit_num] = min(ramping[i],min_and_max[i][0] - state0[i + unit_num],state0[i + 2*unit_num])
            state0[i + 3*unit_num] = min(ramping[i],state0[i + unit_num]-min_and_max[i][1],state0[i + 3*unit_num])
        else:
            state0[i + unit_num] = 0
            state0[i + 2*unit_num] = 0
            state0[i + 3*unit_num] = 0
  
    return state0
def env_s(state,state2,stage,wind,load,load_scene,scene):
    state[4*unit_num] = wind[stage][1]
    state[4*unit_num + 1] = load[stage][0]
    state2[:,0:1] = np.transpose(load_scene[stage + 1:stage+2,:]) 
    state2[:,1:11] = scene
    return state,state2

def init_s(state,state2,stage,wind,load,load_scene,scene):
    state ,state2= env_s(state,state2,stage,wind,load,load_scene,scene)
    g_need = state[4*unit_num+1]-state[4*unit_num]
    min_g = np.sum(min_and_max[:,1:2],)
    max_g = np.sum(min_and_max[:,0:1])
    weight = np.zeros(unit_num)
    if min_g < g_need:
        for i in range(unit_num):
            weight[i] = (min_and_max[i][0]-min_and_max[i][1])/(max_g-min_g)
    for i in range(unit_num):
        state[i] = ini_time[i]
        state[i+unit_num] = min_and_max[i][1]+weight[i]*(g_need-min_g)
        state[i+unit_num*2] = min_and_max[i][0] * 0.05
        state[i+unit_num*3] = min_and_max[i][0] * 0.05
    state = clip_s(state)
    return state,state2

s_dim = 4*unit_num+2
a_dim = unit_num * 3
scene_length = 6
a_bound = 1
state2 = np.zeros((scene_length ,11))
state=np.zeros(s_dim)

td3 =TD3_gru.TD3(a_dim, s_dim, a_bound)
#td3 =DDPG.ddpg(a_dim, s_dim, a_bound)
stage = 56
sess = tf.Session()
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
td3.init_copy()

"""
#train
noise_var = 0.1
r_all = np.zeros(MAX_EPISODES)
for i in range(MAX_EPISODES):
    stage = np.random.randint(6,train_data_hours - scene_length*9)
    scene = sc.scene_choose(stage)
    state,state2 = init_s( state,state2,stage, wind, load, load_scene,scene)
    r_sum = 0
    noise_var = math.exp(-1*3/MAX_EPISODES*i) *0.2 + 0.05
    for j in range(MAX_EP_STEPS):
        action = td3.choose_action(state,state2)
        action_noise = np.clip(np.random.normal(0,noise_var,size=(a_dim,)),-1,1)
        action = np.clip(action+action_noise,-1,1)
        state_ = update_state(state,action)
        scene = sc.scene_choose(stage+1)
        state_,state2_ = env_s(state_,state2, stage+1,wind,load,load_scene,scene)
        r,num = get_reward(state_)
        r_all[i] += r
        r_sum += r
        td3.store_transition(state , action, r, state_,state2,state2_)
        if td3.pointer > 500 and (td3.pointer+1)%6 == 0:
            #td3.learn()
            td3.learn_c()
            if (td3.pointer+1)%4 == 0:
                td3.learn_a()
        state =state_
        state2 = state2_
        stage+=1
  
save_path = saver.save(td3.sess, './IEEE30/model.ckpt')

#3_0 风机接3号机组
print("model saved in path: %s" % save_path, flush=True)
"""


#test
wind,wind_f,wind_r,load,load_scene = tt.train_or_test('test')
load = load*10/4
load_scene = load_scene*10/4
wind[:,0:2] = unit_limit[wind_location][0]*wind[:,0:2]/wind[:,2:3] 
saver.restore(td3.sess, tf.train.latest_checkpoint('./IEEE30_3_2.5'))
state2 = np.zeros((scene_length ,11))
state=np.zeros(s_dim)
stage = 34 + 24*2
stage =stage - stage%24

scene = sc.scene_generation(stage, wind_f,wind_r)
state,state2 = init_s( state,state2,stage, wind, load, load_scene,scene)
ret = 0
error_sum = 0
e_num = 0
r_increase = np.zeros(50*24)
r_sum = np.zeros(50)
time_start0=time.time()

while ret < 50:
#state,state2 = env.init_s( state,state2,stage, wind, load, load_scene,scene)
    days = 24
    time_start=time.time()
    #state,state2 = init_s( state,state2,stage, wind, load, load_scene,scene)
    t = np.zeros(days)
    sum_e = np.zeros(days)
    sum_load = np.zeros(days)
    sum_wind = np.zeros(days)
    sum_up = np.zeros(days)
    sum_down = np.zeros(days)
    wind_p = np.zeros(days)
    q = np.zeros(days)
    unit_state = np.zeros((unit_num,days))
    unit_p = np.zeros((days,unit_num))
    r = 0
    s_num = 0
    for i in range(days):
        r0,num = get_cost(state)
        s_num+=num
        r += r0
        r_increase[ret*24+i] += (-r0)
        if ret*24+i+1 < r_increase.shape[0]:
            r_increase[ret*24+i+1] = r_increase[ret*24+i]
        r_sum [ret] +=  r0
        action = td3.choose_action(state,state2) 
        state_ = update_state(state,action)
        scene = sc.scene_generation(stage+1, wind_f,wind_r)
        state_,state2_ = env_s(state_,state2, stage+1,wind,load,load_scene,scene)
        t[i] = i
        wind_p[i] = (wind[stage][0])*100
        sum_wind[i] = state[4*unit_num]*100
        sum_load [i] = state[4*unit_num + 1] *100
        
        for j in range(unit_num):
            if state[j] > 0:
                unit_state[j][i] = 1
                unit_p[i][j] = state[j+unit_num]*100
            else:
                unit_state[j][i] = 0
                unit_p[i][j] = 0
            sum_e[i] += state[j+unit_num]*100
            sum_up[i] += state[j+2*unit_num] *100
            sum_down[i] += state[j+3*unit_num] *100
        sum_up[i] = min(state[4*unit_num+1]*10,sum_up[i])
        sum_down[i]  = min(state[4*unit_num+1]*10,sum_down[i])
        q[i] = (sum_load[i] - sum_wind[i] - sum_e[i])
        if q[i]>sum_up[i] or (-1*q[i])>sum_down[i]:
            e_num += 1
        error_sum += abs(q[i])
        state =state_
        state2 =state2_
        stage+=1
    
    print(r,s_num)
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
    ret +=1

time_end0=time.time()     
print('time cost',time_end0-time_start0,'s')
print(sum(r_sum))




plt.plot(t,sum_load,color = 'black',label ='load')
p1 = plt.bar(t, sum_e,label = 'unit')  
p2 = plt.bar(t, sum_wind, bottom = sum_e, label = 'wind')  #在p1的基础上绘制，底部数据就是p1的数据
plt.legend()
plt.show()
plt.close()
plt.bar(t,sum_up,label = 'up')
plt.plot(t, q,color = 'black')
plt.bar(t,-sum_down,label = 'down')
plt.legend()
plt.show()
plt.close()
plt.figure(figsize=(10,6),dpi = 300)
plt.plot(t,unit_p[:,0:1],linestyle = '--')
plt.scatter(t,unit_p[:,0:1],label = 'G1',marker='.')
plt.plot(t,unit_p[:,1:2],linestyle = '--')
plt.scatter(t,unit_p[:,1:2],label = 'G2',marker='^')
plt.plot(t,unit_p[:,2:3],linestyle = '--')
plt.scatter(t,unit_p[:,2:3],label = 'G3',marker=',')
plt.plot(t,unit_p[:,3:4],linestyle = '--')
plt.scatter(t,unit_p[:,3:4],label = 'G4',marker='v')
plt.plot(t,unit_p[:,4:5],linestyle = '--')
plt.scatter(t,unit_p[:,4:5],label = 'G5',marker='<')
plt.ylim(0,200)
plt.legend(loc='best',ncol = 3, prop={"family": "Times New Roman", "size": 15})
plt.xlabel('Time period (h)',fontdict={"family": "Times New Roman", "size": 20})
plt.ylabel('Power generation (MW)',fontdict={"family": "Times New Roman", "size": 20})
plt.xticks(fontproperties = 'Times New Roman', size = 15)
plt.yticks(fontproperties = 'Times New Roman', size = 15)
plt.show()
plt.close()
