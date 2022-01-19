# -*- coding: utf-8 -*-
"""
動的最適化問題であるMPB(Moving Peak Benchmaek)を探索するプログラム
群れを分けて探索することで広域の探索が実現でき、変化に対応出来る
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from sklearn.cluster import KMeans
import warnings
import time
import random
from deap.benchmarks import movingpeaks

# jitのwarning用
warnings.simplefilter('ignore')

np.random.seed(2)

# パラメータ設定
N = 60 # 個体数
d = 5 # 次元

lim = 30
xmax = 100
xmin = 0
frequency = 5 * 1000 #問題変化頻度
MaxIT = 50000 # 繰り返す周期
Tv = 0.05
rnd = random.Random()

@jit
# ルーレット選択用関数
def roulette_choice(w):
    t = np.cumsum(w)
    r = np.random.random() * np.sum(w)  #0∼1の乱数*重みの累積和（最大値が累積和になるようにするため）
    for i, e in enumerate(t):   #インデックスと値をenumurateで取得できる間ずっと代入し続ける
        if r < e:
            return i    #累積和が乱数より大きい場合iを返す

@jit
def fit(x):
    
    z = -mp(x[:d],count = False)[0]
    
    if z > 0:
        z = 1 / (1 + z)
    else:
        z = 1 + abs(z)
        
    return z

@jit
def x_check(x):
    if x > xmax:
        x = xmax
    elif x < xmin:
        x = xmin
    
    return x

def cul_offline():
    
    off = mp.currentError()
    error.append(off)
    
    error_basic.append(off2)
    
    

# ABCアルゴリズム
def ABC(x,n):
    
    # employee bee step
    for i in range(n):
        v = x.copy()
        k = i
        while k == i:
            k = np.random.randint(n)
        
        r = np.random.rand()*2-1 #-1から1までの一様乱数
        j = np.random.randint(d) #変更する次元
        v[i,j] = x[i,j] + r * (x[i,j] - x[k,j]) #近傍点計算
        v[i,j] = x_check(v[i,j])
        if fit(x[i]) < fit(v[i]):
            x[i] = v[i]
            x[i,-1] = 0
        else: x[i,-1] += 1

    # onlooker bee step
    for i in range(n):
        v = x.copy()
        w = []
        for j in range(n):
            w.append(fit(x[j]))
        l = roulette_choice(w)
        
        k = l
        while k == l:
            k = np.random.randint(n)
        
        r = np.random.rand()*2-1 #-1から1までの一様乱数
        j = np.random.randint(d)
        v[l,j] = x[l,j] + r * (x[l,j] - x[k,j])
        v[l,j] = x_check(v[l,j])
        if fit(x[l]) < fit(v[l]):
            x[l] = v[l]
            x[l,-1] = 0
        else: x[l,-1] += 1

    # scout bee step
    i = np.argmax(x,axis = 0)[-1]
    if x[i,-1] > lim:
        x[i,:d] = np.random.rand(d)*(xmax-xmin) + xmin
        x[i,-1] = 0

    return x

@jit
def update_before(x,sub,m):
    
    if sub * m > len(x): # 個体数が割り切れないとき
        new = np.zeros(((sub * m) - len(x),d + 1)) #足りない分追加
        for i in range(len(new)): #個体数調整
            new[i][:d] = (xmax-xmin)*np.random.rand(d) + xmin #解の初期化
            new[i][-1] = 0
        x = np.append(x,new,axis = 0)
            
    else:
        # 個体数調整
        while len(x) > N:
            x = np.delete(x,0,axis = 0)
            
    return x


def update_after(x,m,best_value):
    
    best_after = mp(x[0][:d],count = False)[0]
    for i in range(N):
        a = x[i][:d]
        if best_after < mp(a,count = False)[0]:
            best_after = mp(a,count = False)[0]
            
    cs = best_value - best_after
    
    #print(best_before)
    #print(best_after)
    #print(cs)
    #print(it)
    if cs > Tv: #定数と比較してサブ集合数を変更
        m += 1
    elif m > 2:
        m -= 1
    
    
    # クリアリングスキーム
    unique = [] #ユニークな要素があるリスト
    unique_level = 0
    #print(f"ユニーク{len(np.unique(np.round(x[:,:d],unique_level),axis = 0))}")
    for u in np.unique(np.round(x[:,:d],unique_level),axis = 0): #ユニークな要素を検索
        unique.append(list(zip(*np.where(np.round(x[:,:d],unique_level) == u))))
    new_x = np.zeros((len(unique),d + 1)) #新しい母集団を定義
    for i in range(len(unique)): #ユニークな母集団を定義
        new_x[i] = x[unique[i][0][0]]

    x = new_x.copy() #母集団を定義
    #print(len(x))
    if len(x) < N:
        new = np.zeros((N - len(x),d + 1))
        for i in range(len(new)):
            new[i][:d] = (xmax-xmin)*np.random.rand(d) + xmin
            new[i][-1] = 0
            x = np.append(x,new,axis = 0) #解の置き直し
    
    np.random.shuffle(x)
    
    return x,m,cs
    

# 結果出力用リスト
offline_error = []
offline_basic = []
distance_result = []
distance2_result = []
center_list = []
center2_list = []

#時間計測
start_time = time.time()

# 実験のループ
for loup in range(10):
    # the moving peak benchmark
    m = 2 #群れの数
    m2 = 1
    update = True
    state = 1
    cs = -1
    
    # the moving peak benchmark
    sc = movingpeaks.SCENARIO_2
    sc["lambda"] = 0
    sc["move_severity"] = 1
    sc["uniform_height"] = 50
    sc["uniform_width"] = 1
    sc["npeaks"] = 20
    
    mp = movingpeaks.MovingPeaks(dim=d, random=rnd, **sc)
    maximum = mp.globalMaximum()[0] 
    
    error = []    # 誤差リスト
    error_basic = []
    distance_list = []
    distance_2list = []
    center_std = []
    center2_std = []
    
    x = (xmax-xmin)*np.random.rand(N,d) + xmin #解の初期化
    x = np.concatenate([x,np.zeros((N,1))],axis = 1)
    x_best = x[0][:d] #x_bestの初期化
    
    x_2 = x.copy()
    x_2best = x_2[0][:d]
    
    best_before = mp(x[0][:d],count = False)[0]
    best2 = mp(x_2[0][:d],count = False)[0]
    best_value = 0
    for i in range(N):
        a = x[i][:d]
        if best_before < mp(a,count = False)[0]:
            best_before = mp(a,count = False)[0]
            x_best = a.copy()
    
    best_ratio = []
    do = 0
    # アルゴリズム
    while state < 10 : #itが繰り返し回数以下とbestが最大未達の時実行
        do += 1
        #print(do)
        #print(best_before)
        
        if update:
            
            sub = int(N / m) # サブ集合数
            sub2 = int(N / m2)
            
            x = update_before(x,sub,m)
            
            #x_2 = update_before(x_2,sub2,m2)
            
            """
            if cs > Tv:
                extra = (xmax - xmin) * np.random.rand(sub,d) + xmin
                extra = np.concatenate([extra,np.zeros((sub,1))],axis = 1)
                x[0:sub] = extra
            """
        
            best_before = mp(x[0][:d],count = False)[0]
            best2 = mp(x_2[0][:d],count = False)[0]
            best_value = 0
            for i in range(N):
                a = x[i][:d]
                if best_before < mp(a,count = False)[0]:
                    best_before = mp(a,count = False)[0]
                    x_best = a.copy()
                    
            maximum = mp.globalMaximum()[0]
            
            update = False

        # サブポピュレーションでサーチ
        for i in range(m2):
            x_2[i * sub2:(i + 1) * sub2] = ABC(x_2[i * sub2:(i + 1) * sub2],sub2)#サブ集合に分割
            
            population = x_2[i * sub2:(i + 1) * sub2]
            for j in population:
                a = j[:d]
                if best2 < mp(a,count = False)[0]:
                    best2 = mp(a,count = False)[0]
                    off2 = maximum - best2
                    x_2best = a.copy()
        
        for i in range(m):
            if update == False:
                x[i * sub:(i + 1) * sub] = ABC(x[i * sub:(i + 1) * sub],sub)#サブ集合に分割
            
                #csを算出
                population = x[i * sub:(i + 1) * sub]
                for j in population:
                    a = j[:d]
                    b = mp(a)[0]
                    if b != mp(a,count = False)[0]:
                        update = True
                        cul_offline()
                        maximum = mp.globalMaximum()[0] 
                        #print(maximum)
                        state += 1
                        break
                    if best_before < b:
                        best_before = b
                        x_best = a
        
        # 世代進展処理    
        if update:
        
            """
            model = KMeans(n_clusters = m)
            model.fit(x[:,:d])
            ans = model.labels_
            center = []
            for c in range(m):
                cluster_list = x[np.where(ans == c)] #クラスターリスト作成
                center.append(np.mean(cluster_list[:,:d],axis=0)) #クラスター中心算出
                center_std.append(np.round(np.std(center,axis = 0),4))           
            
            model = KMeans(n_clusters = m)
            model.fit(x_2[:,:d])
            ans = model.labels_
            center = []
            for c in range(m):
                cluster_list = x_2[np.where(ans == c)] #クラスターリスト作成
                center.append(np.mean(cluster_list[:,:d],axis = 0)) #クラスター中心算出
                center2_std.append(np.round(np.std(center,axis = 0),4))
            """
        
            #x_2,m2,_ = update_after(x_2,m2,best_value)
            x,m,cs = update_after(x,m,best_before)
            
            #print(state)
    
    # 出力
    print(f"ループ{loup}回目")  
    print(f"最良解は、{x_best}です")
    print(f"最良値は、{best_before}です")
    print(f"エラー改良{np.round(error,4)}です")
    print(f"エラー{np.round(error_basic,4)}です")
    print(f"群れの数は、{m}です")
    print(m2)
    #print(f"距離平均_alter{np.mean(distance_list)}")
    #print(f"距離平均_Multi{np.mean(distance_2list)}")
    offline_error.append(np.mean(error))
    offline_basic.append(np.mean(error_basic))
    distance_result.append(np.mean(distance_list))
    distance2_result.append(np.mean(distance_2list))
    center_list.append(np.mean(center_std,axis = 0))
    center2_list.append(np.mean(center2_std,axis = 0))
    
#実験出力
elapsed_time = time.time() - start_time
print(f"所要時間は、{elapsed_time}です")
print("alter:")
print(f"エラーの値は、{np.round(offline_error,4)}です。")
error_ave = sum(offline_error) / len(offline_error)
print(f"エラー平均は、{np.round(error_ave,4)}です。")
error_std = (sum((offline_error - error_ave) ** 2) /len(offline_error)) ** 0.5
print(f"エラー標準偏差は、{np.round(error_std,4)}です。") 
print(f"距離は、{np.mean(distance_result)}です")
print(f"クラスター標準偏差{np.mean(center_list,axis = 0)}です。")
print("Multi:")
print(f"エラーの値は、{np.round(offline_basic,4)}です。")
error_ave = sum(offline_basic) / len(offline_basic)
print(f"エラー平均は、{np.round(error_ave,4)}です。")
error_std = (sum((offline_basic - error_ave) ** 2) /len(offline_basic)) ** 0.5
print(f"エラー標準偏差は、{np.round(error_std,4)}です。") 
print(f"距離は、{np.mean(distance2_result)}です")
print(f"クラスター標準偏差{np.mean(center2_list,axis = 0)}です。")