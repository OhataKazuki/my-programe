# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 00:41:20 2022

@author: kazu
"""


#ライブラリのインポート
from control.matlab import *
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import warnings
import time

# jitのwarning用
warnings.simplefilter('ignore')

#seed値の決定
np.random.seed(9)

#パラメータの設定
s = 40
c1 = 1.494
c2 = 1.494
w = 0.729

lim = 20
TC = np.zeros(s) #更新カウント

xmin = 0
xmax = 200
d = 3

initial = (xmax - xmin) / 2
delta = (xmax - xmin) / 10
cd = delta / 10
nl = 5

#ボード線図表示用の設定
plt.rcParams['font.family'] ='sans-serif' #使用するフォント
plt.rcParams['xtick.direction'] = 'in' #x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in' #y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.major.width'] = 1.0 #x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0 #y軸主目盛り線の線幅
plt.rcParams['font.size'] = 10 #フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0 # 軸の線幅edge linewidth。囲みの太さ
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.xmargin'] = '0' #'.05'
plt.rcParams['axes.ymargin'] = '0.05'
plt.rcParams['savefig.facecolor'] = 'None'
plt.rcParams['savefig.edgecolor'] = 'None'

#ボード線図表示用の設定
def linestyle_generator():
    linestyle = ['-', '--', '-.', ':']
    lineID = 0
    while True:
        yield linestyle[lineID]
        lineID = (lineID + 1) % len(linestyle)

#図示
def plot_set(fig_ax, *args):
    fig_ax.set_xlabel(args[0])
    fig_ax.set_ylabel(args[1])
    fig_ax.grid(ls=':')
    if len(args)==3:
        fig_ax.legend(loc=args[2])

#ボード線図の図示
def bodeplot_set(fig_ax, *args):
    fig_ax[0].grid(which="both", ls=':')
    fig_ax[0].set_ylabel('Gain [dB]')

    fig_ax[1].grid(which="both", ls=':')
    fig_ax[1].set_xlabel('$\omega$ [rad/s]')
    fig_ax[1].set_ylabel('Phase [deg]')
    
    if len(args) > 0:
        fig_ax[1].legend(loc=args[0])
    if len(args) > 1:
        fig_ax[0].legend(loc=args[1])

#制御工学の評価指標計算式
@jit
def ISE(x):
    K = tf([x[0], x[1], x[2]], [1, 0])
    Gyr = feedback(P*K, 1)
    y, t = step(Gyr, np.arange(0, 10, 0.05))

    func = 0
    for i in y:
        func += (i - 1)**2
        
    return func

def IAE(x):
    K = tf([x[0], x[1], x[2]], [1, 0])
    Gyr = feedback(P*K, 1)
    y, t = step(Gyr, np.arange(0, 10, 0.05))

    func = 0
    for i in y:
        func += np.abs(i - 1)
        
    return func

def ITAE(x):
    K = tf([x[0], x[1], x[2]], [1, 0])
    Gyr = feedback(P*K, 1)
    y, t = step(Gyr, np.arange(0, 10, 0.05))

    func = 0
    for i,num in enumerate(y):
        func += i*(num - 1)
        
    return func

def ITSE(x):
    K = tf([x[0], x[1], x[2]], [1, 0])
    Gyr = feedback(P*K, 1)
    y, t = step(Gyr, np.arange(0, 10, 0.05))

    func = 0
    for i,num in enumerate(y):
        func += i* ((num - 1)**2)
        
    return func

#限界感度法
@jit
def ZN(P):
    num_delay, den_delay = pade( 0.005, 1)
    Pdelay = P * tf(num_delay, den_delay)
    ok = False
    kp0 = 100
    T0 = 1.3
    span = 0.01
    
    while not(ok):
        K = tf([0, kp0], [0, 1])
        Gyr = feedback(Pdelay*K, 1)
        y,t = step(Gyr, np.arange(0, 5, span))
        y2,t2 = step(Gyr, np.arange(0, 10, span))
        if (y.sum()) * 2 > y2.sum():
            ok = True
        else:
            kp0 += 0.1
    
    kp = 0.6 * kp0
    ki = kp / (0.5 * T0)
    kd = kp * (0.125 * T0)
    
    return [kp, ki, kd]

#Particle swarm optimization
def PSO(x):
    
    global velocities, personal_best_positions, personal_best_scores, global_best_particle_id, global_best_particle_position

    for i in range(s):
        r1 = np.random.rand()
        r2 = np.random.rand()
        velocities[i] = velocities[i] * w + r1 * c1 * (personal_best_positions[i] - x[i]) + r2 * c2 * (global_best_particle_position - x[i])
        x[i] += velocities[i]
            
        score = ISE(x[i])
        if score < personal_best_scores[i]:
            personal_best_scores[i] = score
            personal_best_positions[i] = x[i]

        # グローバルベストの更新
        global_best_particle_id = np.argmin(personal_best_scores)
        global_best_particle_position = personal_best_positions[global_best_particle_id]
        
    return x

@jit
# ルーレット選択用関数
def roulette_choice(w):
    t = np.cumsum(w)
    r = np.random.random() * np.sum(w)  #0∼1の乱数*重みの累積和（最大値が累積和になるようにするため）
    for i, e in enumerate(t):   #インデックスと値をenumurateで取得できる間ずっと代入し続ける
        if r < e:
            return i    #累積和が乱数より大きい場合iを返す

#適応度関数
@jit
def fit(x):
    
    z = ISE(x)
    
    if z > 0:
        z = 1 / (1 + z)
    else:
        z = 1 + abs(z)
    
    return z

#定義域に解を戻す関数
@jit
def x_check(x):
    if x > xmax:
        x = xmax
    elif x < xmin:
        x = xmin
    
    return x

@jit
# ABCアルゴリズム
def ABC(x):
    
    # employee bee step
    for i in range(s):
        v = x.copy()
        k = i
        while k == i:
            k = np.random.randint(s)
        
        r = np.random.rand()*2-1 #-1から1までの一様乱数
        j = np.random.randint(d) #変更する次元
        v[i,j] = x[i,j] + r * (x[i,j] - x[k,j]) #近傍点計算
        v[i,j] = x_check(v[i,j])
        
        if fit(x[i]) < fit(v[i]):
            x[i] = v[i]
            TC[i] = 0
        else: TC[i] += 1

    # onlooker bee step
    for i in range(s):
        v = x.copy()
        w = []
        for j in range(s):
            w.append(fit(x[j]))
        l = roulette_choice(w)
        
        k = l
        while k == l:
            k = np.random.randint(s)
        
        r = np.random.rand()*2-1 #-1から1までの一様乱数
        j = np.random.randint(d) #変更する次元
        v[i,j] = x[i,j] + r * (x[i,j] - x[k,j]) #近傍点計算
        v[i,j] = x_check(v[i,j])
        
        
        if fit(x[l]) < fit(v[l]):
            x[l] = v[l]
            TC[l] = 0
        else: TC[l] += 1

    # scout bee step
    i = np.argmax(TC)
    if TC[i] > lim:
        x[i] = (xmax - xmin) * np.random.rand(d)
        TC[i] = 0
        
    return x

# ABC_LISアルゴリズム
def ABC_LIS(x):
    
    # employee bee step
    for i in range(s):
        v = x.copy()
        r = np.random.rand()*2-1 #-1から1までの一様乱数
        j = np.random.randint(d) #変更する次元
        v[i,j] = x[i,j] + (r * x[i,-2])
        k_list = []
        for k in range(s):
            if i != k:
                diff = np.sqrt(np.sum((x[i,:d] - x[k,:d])**2))
                if diff < x[i,-2]:
                    k_list.append(k)
                    
        if len(k_list) > 0:
            k = np.random.choice(k_list)
            v[i,j] = x[i,j] + r * (x[i,j] - x[k,j])
                    
        v[i,j] = x_check(v[i,j])
        if fit(x[i]) < fit(v[i]):
            x[i] = v[i].copy()

    # onlooker bee step
    for i in range(s):
        v = x.copy()
        w = []
        
        for j in range(s):
            for k in range(s):
                if k != j:
                    diff = np.sqrt(np.sum((x[j,:d] - x[k,:d])**2))
                    if diff < x[j,-2]:
                        w.append(fit(x[j]))
                        break
                    elif k == s - 1:
                        w.append(0)
        
        l = roulette_choice(w)
        
        if l is not None:
            r = np.random.rand()*2-1 #-1から1までの一様乱数
            j = np.random.randint(d)
            v[l,j] = x[l,j] + (r * x[l,-2]) 
            k_list = []
            for k in range(s):
                if l != k:
                    diff = np.sqrt(np.sum((x[l,:d] - x[k,:d])**2))
                    if diff < x[l,-2]:
                        k_list.append(k)
                        
            if len(k_list) > 0:
                k = np.random.choice(k_list)
                v[l,j] = x[l,j] + r * (x[l,j] - x[k,j]) #近傍点計算
                        
                    
            v[l,j] = x_check(v[l,j])
            if fit(x[l]) < fit(v[l]):
                x[l] = v[l].copy()

    # scout bee step
    x[:,-1] = 1
    scout = 0
    for i in range(s):
        for j in range(s):
            if i != j:
                diff = np.sqrt(np.sum((x[i,:d] - x[j,:d])**2))
                if diff < x[i,-2]:
                    if fit(x[i]) <= fit(x[j]):
                        x[i,-1] += 1
                    
        rate = 1 / (1 + np.exp(2 * (nl - x[i,-1])))
        r = np.random.rand()
        if rate > r:
            x[i,:d] = (xmax-xmin)*np.random.rand(d) + xmin
            x[i,-2] = initial
            scout += 1
    
    """
    #更新
    for i in range(s):
        if x[i,-1] == 1:
            flug = True
            add = True
            for j in range(s):
                if i != j:
                    diff = np.sqrt(np.sum((x[i,:d] - x[j,:d])**2))
                    if (diff + cd) < x[i,-2]:
                        if x[i,-2] > delta:
                            x[i,-2] -= delta
                        flug = False
                        break
                    elif (diff + cd) < (x[i, -2] + delta):
                        add = False
                        break
            if flug and add:
                x[i,-2] += delta
        else:
            flug = True
            for j in range(s):
                if i != j:
                    diff = np.sqrt(np.sum((x[i,:d] - x[j,:d])**2))
                    if diff < cd:
                        if x[j,-1] == 1:
                            x[i,-2] = x[j,-2]
                            flug = False
            if flug:
                x[i,-2] += delta
    """
    
    
    return x

#探索を行うプログラム
x = np.zeros((s,d))
for i in range(s):
    x[i] = (xmax - xmin) * np.random.rand(d)
    
x = np.concatenate([x,(initial * np.ones((s,1)))],axis = 1)
x = np.concatenate([x,np.ones((s,1))],axis = 1)

P = tf( [0,1], [1, 9, 23, 15] )

# 各粒子の速度
velocities = np.zeros((s,d))

# 各粒子ごとのパーソナルベスト位置
personal_best_positions = np.copy(x[:,:d])

# 各粒子ごとのパーソナルベストの値
personal_best_scores = np.ones(s)

# グローバルベストの粒子ID
global_best_particle_id = np.argmin(personal_best_scores)

# グローバルベスト位置
global_best_particle_position = personal_best_positions[global_best_particle_id]

best_score = fit(global_best_particle_position)

ZN_errors = []
errors = []

#探索
for i in range(50):
    
    bottom = np.abs([1, 9 + np.random.randn(), 23 + np.random.randn(), 15 + np.random.randn()])
    P = tf([0,1], bottom)
    ZN_gain = ZN(P)
    
    kp = np.round(ZN_gain[0], 1)
    ki = np.round(ZN_gain[1], 1)
    kd = np.round(ZN_gain[2], 1)
    
    x = PSO(x[:,:d])
    
    #x = ABC(x[:,:d])
    #x = ABC_LIS(x)
    """
    for j in range(s):
        if fit(x[j]) > best_score:
            best_score = fit(x[j])
            global_best_particle_position = x[j].copy()
    """
    
    #誤差を計算     
    ZN_errors.append(ISE([kp,ki,kd]))
    errors.append(ISE(global_best_particle_position))              
    
LS = linestyle_generator()
fig, ax = plt.subplots(figsize=(3, 2.3))
"""
kp = 115.2
ki = 175.9
kd = 18.9

print(ISE(global_best_particle_position))
K = tf([global_best_particle_position[0], global_best_particle_position[1], global_best_particle_position[2]], [1, 0])
Gyr = feedback(P*K, 1)
y, t = step(Gyr, np.arange(0, 2, 0.01))
ax.plot(t, y, label = "swarm")

print(ISE([kp,ki,kd]))
K = tf([kp, ki, kd], [1, 0])
Gyr = feedback(P*K, 1)
y, t = step(Gyr, np.arange(0, 2, 0.01))
ax.plot(t, y, label = "ZH")

ax.axhline(1, color="k", linewidth=0.5) 
plot_set(ax, 't', 'y', 'upper left')
 
ax.set_xlim(0, 2)
ax.set_ylim(0,2)
plt.show()
"""

print("-------------")
print("原価感度法の誤差")
print(np.mean(ZN_errors))
print("群知能での誤差")
print(np.mean(errors))