# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 00:26:34 2021

@author: kazu

このプログラムは、動的最適化問題にALISを加えたABCアルゴリズムを適用したプログラムです。
ALISとは、群れの探索性を維持して環境の変化に対応できるように設計されたメカニズムです。
"""

# ライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import warnings
import time
import random

# jitのwarning用
warnings.simplefilter('ignore')

#乱数シードの固定
np.random.seed(0)

# パラメータ設定
N = 60 # 個体数
d = 5 # 次元
lim = 30

#　問題設定
xmax = 100
xmin = 0
frequency = 5 * 1000 #問題変化頻度
MaxIT = 50000 # 繰り返す周期
Tv = 0.05
rnd = random.Random()

#ALISの設定
initial = (xmax - xmin) / 2
delta = (xmax - xmin) / 10
cd = delta / 10
nl = 5

@jit
# ルーレット選択用関数
def roulette_choice(w):
    t = np.cumsum(w)
    r = np.random.random() * np.sum(w)  #0∼1の乱数*重みの累積和（最大値が累積和になるようにするため）
    for i, e in enumerate(t):   #インデックスと値をenumurateで取得できる間ずっと代入し続ける
        if r < e:
            return i    #累積和が乱数より大きい場合iを返す

# moving peak benchmark(MPB)
def change_function():
    
    #　グローバル変数導入
    global it,v,H,W,X,F

    # 頂点の移動 
    r = np.random.rand(p,d) -0.5
    v = S * ((1 - ramda) * r + ramda * v) / np.abs(r + v)
    #v = 0
    X = X + (v / 1)
    X[X > xmax] = xmax
    X[X < xmin] = xmin
    #delta1 = np.random.normal(0.5,(0.5/3),p)
    #delta2 = np.random.normal(0.5,(0.5/3),p)
    delta1 = np.random.randn(p)
    delta2 = np.random.randn(p)
    H = H + (hs * (delta1 / 1))
    for i in range(p):
        if H[i] > hmax:
            H[i] = (2 * hmax) - H[i]
        elif H[i] < hmin:
            H[i] = (2 * hmin) - H[i]
    W = W + (ws * (delta2/1))
    for i in range(p):
        if W[i] > wmax:
            W[i] = (2 * wmax) - W[i]
        elif W[i] < wmin:
            W[i] = (2 * wmin) - W[i]
    
    it += 1

@jit
# the moving peak
def func(x):   
            
    #　関数値
    F = H - W * np.sqrt(np.sum((np.tile(x,[p,1]) - X) ** 2,axis = 1))
    f = np.max(F)
        
    return f

# 適応度関数
@jit
def fit(x):
    
    z = -func(x[:d])
    
    if z > 0:
        z = 1 / (1 + z)
    else:
        z = 1 + abs(z)
        
    return z

# 制約確認関数
@jit
def x_check(x):
    if x > xmax:
        x = xmax
    elif x < xmin:
        x = xmin
    
    return x

#　誤差計算関数
@jit
def cul_offline():
    
    off = np.max(H) - best_before
    error.append(off)
    
    #print(f"最適頂点に{peak}")


# ALIS　ABCアルゴリズム
def ABC_multi(x,best):
    
    # employee bee step
    for i in range(N):
        v = x.copy()
        r = np.random.rand()*2-1 #-1から1までの一様乱数
        j = np.random.randint(d) #変更する次元
        v[i,j] = x[i,j] + (r * x[i,-2])
        k_list = []
        for k in range(N):
            if i != k:
                diff = np.sqrt(np.sum((x[i,:d] - x[k,:d])**2))
                if diff < x[i,-2]:
                    k_list.append(k)
                    
        if len(k_list) > 0:
            k = np.random.choice(k_list)
            v[i,j] = x[i,j] + r * (x[i,j] - x[k,j])
                    
        v[i,j] = x_check(v[i,j])
        if fit(x[i]) < fit(v[i]):
            x[i] = v[i]

    # onlooker bee step
    for i in range(N):
        v = x.copy()
        w = []
        
        for j in range(N):
            for k in range(N):
                if k != j:
                    diff = np.sqrt(np.sum((x[j,:d] - x[k,:d])**2))
                    if diff < x[j,-2]:
                        w.append(fit(x[j]))
                        break
                    elif k == N - 1:
                        w.append(0)
        
        l = roulette_choice(w)
        
        if l is not None:
            r = np.random.rand()*2-1 #-1から1までの一様乱数
            j = np.random.randint(d)
            v[l,j] = x[l,j] + (r * x[l,-2])
            k_list = []
            for k in range(N):
                if l != k:
                    diff = np.sqrt(np.sum((x[l,:d] - x[k,:d])**2))
                    if diff < x[l,-2]:
                        k_list.append(k)
                        
            if len(k_list) > 0:
                k = np.random.choice(k_list)
                v[l,j] = x[l,j] + r * (x[l,j] - x[k,j]) #近傍点計算
                        
                    
            v[l,j] = x_check(v[l,j])
            if fit(x[l]) < fit(v[l]):
                x[l] = v[l]

    # scout bee step
    x[:,-1] = 1
    scout = 0
    for i in range(N):
        for j in range(N):
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
    
    #移動距離更新
    for i in range(N):
        if x[i,-1] == 1:
            flug = True
            add = True
            for j in range(N):
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
            for j in range(N):
                if i != j:
                    diff = np.sqrt(np.sum((x[i,:d] - x[j,:d])**2))
                    if diff < cd:
                        if x[j,-1] == 1:
                            x[i,-2] = x[j,-2]
                            flug = False
            if flug:
                x[i,-2] += delta
    
    
    return x



# 結果出力用リスト
offline_error = []
offline_basic = []
distance_result = []
center_list = []

#時間計測
start_time = time.time()

# 実験のループ
for loup in range(10):
    # ループ回数またはフラグ等
    update = True
    state = 1
    it = 0
    
    # the moving peak benchmark
    p = 10 #頂点の個数
    hmax = 70
    hmin = 30
    wmax = 12
    wmin = 1
    hs = 7 #高さの変化強度
    ws = 1 #幅の変化強度
    S = 1 #移動距離
    ramda = 0 #共通定数
    H =50 * np.ones(p)
    W = np.ones(p)
    F = np.random.rand(p)
    v = 0
    width = (xmax - xmin) / p
    X = (xmax - xmin) * np.random.rand(p,d) + xmin
    change_function()
    
    #　結果出力用リスト
    error = []
    distance_list = []
    center_std = []
    
    #解の初期化
    x = (xmax-xmin)*np.random.rand(N,d) + xmin 
    x = np.concatenate([x,(initial * np.ones((N,1)))],axis = 1)
    x = np.concatenate([x,np.ones((N,1))],axis = 1)
    x_best = x[0][:d] #x_bestの初期化
    
    best_ratio = []
    # アルゴリズム
    while state < 10 : #itが繰り返し回数以下とbestが最大未達の時実行

        if update:
            best_before = func(x[0][:d])
            update = False
                
        # 探索
        x = ABC_multi(x,x_best)
    
        #最良解を更新
        for j in x:
            it += 1
            a = j[:d]
            b = func(a)
            if best_before < b:
                best_before = b
                x_best = a.copy()
        
        #問題に変化を起こす分岐
        if it > state * frequency:
            cul_offline()
            change_function()
            state += 1
            update = True
    
    # 出力
    print(f"ループ{loup}回目")  
    print(f"最良解は、{x_best}です")
    print(f"最良値は、{best_before}です")
    print(f"エラー改良{np.round(error,4)}です")
    offline_error.append(np.mean(error))
    distance_result.append(np.mean(distance_list))
    center_list.append(np.mean(center_std,axis = 0))
    
#実験出力
elapsed_time = time.time() - start_time
print(f"所要時間は、{elapsed_time}です")
print(f"エラーの値は、{np.round(offline_error,4)}です。")
error_ave = sum(offline_error) / len(offline_error)
print(f"エラー平均は、{np.round(error_ave,4)}です。")
error_std = (sum((offline_error - error_ave) ** 2) /len(offline_error)) ** 0.5
print(f"エラー標準偏差は、{np.round(error_std,4)}です。") 
print(f"距離は、{np.mean(distance_result)}です")
print(f"クラスター標準偏差{np.mean(center_list,axis = 0)}です。")