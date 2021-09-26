# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 11:35:39 2020

@author: kazu

このプログラムは、連続最適化のABCで搬送経路最適化問題を解くプログラムです。
離散値を確率に写像させて01を反転させる(kmeans-transition)
"""

# ライブラリのインポート
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import itertools
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt
from numba import jit
import warnings
warnings.filterwarnings('ignore') 

customer_count = 12 #顧客数（id=0はdepot）
vehicle_capacity = 40 #車両容量
border = 0.5 #確率定数

np.random.seed(seed=3) 
n = 4 #　クラスターの数

#各顧客のx,y座標をDataFrameとして作成
df = pd.DataFrame({"x":np.random.randint(0,100, customer_count), 
                   "y":np.random.randint(0, 100, customer_count), 
                   "demand":np.random.randint(5, 20, customer_count)})

#id=0はdepotなので，demand=0にする
df.iloc[0].x = 50
df.iloc[0].y = 50
df.iloc[0].demand = 0

#コストとしてノード間の直線距離を求める
cost = pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index).values

df.head()

#kmeansのオブジェクトインスタンス作成
model = KMeans(n_clusters = n)

# 需要の総和を計算
demand_sum = 0
for y in range(customer_count):
    demand_sum += df.iloc[y].demand
#車両数  
vehicle_count = int(demand_sum / vehicle_capacity) + 1 

#ルーレットチョイス
@jit
def roulette_choice(w): 
    t = []
    S = 0
    for e in w:
        S += e
        t.append(S)

    r = np.random.random() * S
    for i, e in enumerate(t):
        if r <= e:
            return i

@jit
#xに０または1を入れて初期化
def generate():
    x = np.zeros((vehicle_count,customer_count,customer_count))
    check = list(range(1,customer_count)) #未達顧客リスト
    
    # 車両ごとのルート採択
    for k in range(vehicle_count):
        capacity = 0
        i = 0 #iを初期化
        while capacity < vehicle_capacity:
            if len(check) > 0:
                    # ランダムで選択
                r = int(np.random.random() * len(check))
                j = check[r]
                if capacity + df.demand[j] < vehicle_capacity:
                    #　移動確定
                    x[k][i][j] = 1
                    capacity += df.demand[j]
                    del check[r]
                    i = j
                else:
                    # 移動終了
                    capacity = capacity + 1              
            else:
                capacity = vehicle_capacity #無限ループ回避
        x[k][i][0] = 1 #depotに戻る
    # 各ルート最適化
    initialize(x)
    
    return x

@jit
# 引数ありのgenerate関数(解の追加用)
def make_route(v):
    check = list(range(1,customer_count))
    
    #check確認
    for k in range(vehicle_count):
        for i in range(customer_count): 
            for j in range(customer_count):
                if v[k][i][j] == 1:
                    if j != 0:
                        check.remove(j)
    
    for k in range(vehicle_count):   
        capacity = 0
        i = 0 #iを初期化
        while capacity < vehicle_capacity:
            if len(check) > 0:
                    # ランダムで選択
                r = int(np.random.random() * len(check))
                j = check[r]
                if capacity + df.demand[j] < vehicle_capacity:
                    #　移動確定
                    v[k][i][j] = 1
                    capacity += df.demand[j]
                    del check[r]
                    i = j
                else:
                    # 移動終了
                    capacity = capacity + 1              
            else:
                capacity = vehicle_capacity #無限ループ回避
        v[k][i][0] = 1 #depotに戻る
    # 各ルート最適化
    initialize(v)
    return v

@jit
# ルート最適化関数
def update(x):
    check = list(range(1,customer_count))
    
    #check確認
    for k in range(vehicle_count):
        for i in range(customer_count): 
            for j in range(customer_count):
                if x[k][i][j] == 1:
                    if j != 0:
                        check.remove(j)
                         #デバッグ用
                    
    for k in range(vehicle_count):
        capacity = 0
        i = 0 #iを初期化
        while capacity < vehicle_capacity:
            if len(check) > 0:
                close = 10000
                for to in range(len(check)):
                    distance = (df.x[check[to]] - df.x[i]) ^ 2 + (df.y[check[to]] - df.y[i]) ^ 2
                    # 近くでまだ行ってない顧客でかつ容量が大丈夫のとき
                    if close > distance:
                        if capacity + df.demand[check[to]] < vehicle_capacity:
                            # 行先決定
                            j = check[to]
                            close = distance

                if i != j:
                    # 移動確定
                    x[k][i][j] = 1
                    capacity += df.demand[j]
                    check.remove(j)
                    i = j
                else:
                    capacity = vehicle_capacity   
            else:
                capacity = vehicle_capacity
        x[k][i][0] = 1 #depotに戻る
    return x

@jit
# 各車両のルート最適化関数
def initialize(x):
    for k in range(vehicle_count):
        x[k] = 0
        update(x)
    return x
    
# 条件に合うかチェックする関数
@jit
def repaie(x):
         
    # 制約 各顧客の場所に訪れるのは1台の車両で1度である
    for j in range(1, customer_count):
        e = 0
        a = []
        for i in range(customer_count):
            for k in range(vehicle_count):
                e = e + x[k][i][j]
                if e != 0: 
                    if x[k][i][j] != 0:
                        a.append([i,k])
        if e > 1:
            return 1
        if e == 0: #顧客の抜けを防止
            return 1 #条件エラー
        
    #depotから出発して，depotに戻ってくる 
    # デポを出発した運搬車が必ず 1つの顧客から訪問を開始することを保証する制約条件    
    for k in range(vehicle_count):
        e = 0
        for j in range(1, customer_count):
            e = e + x[k][0][j]
        if e > 1:
            return 1 #条件エラー
       
        # 必ず 1 つの顧客から運搬車がデポへ到着すること保証する制約条件
        e = 0
        for i in range(1,customer_count):
            e = e + x[0][i][j]
        if e > 1:
            return 1 #条件エラー
         

    #各車両において最大容量を超えない
    for k in range(vehicle_count):
        e = 0
        for i in range(customer_count):
            for j in range(customer_count):
                e = np.sum(df.demand[j] * x[k][i][j])
        if e > vehicle_capacity: #kの要素だけ新たにする
            return 1 #条件エラー
     
        
#部分巡回路除去制約
    subtours = []
    for i in range(2,customer_count):
         subtours += itertools.combinations(range(1,customer_count), i)
    for s in subtours:
        e = 0
        a = []
        for i, j in itertools.permutations(s,2):
            for k in range(vehicle_count):
                e = e + x[k][i][j]
                if e >len(s) - 1:
                    a.append([k,i,j])
        if e >len(s) - 1:
            for h in a:
                x[a[h][0]][a[h][1]][a[h][2]] = 0
                update(x)
    return 0 #条件クリア

#図示する関数
def plot(x):
    
    plt.figure(figsize=(5,5))
    for i in range(customer_count):    
        if i == 0:
            plt.scatter(df.x[i], df.y[i], c='r')
            plt.text(df.x[i]+1, df.y[i]+1, str(i)+",depot")
        else:
            plt.scatter(df.x[i], df.y[i], c='b')
            plt.text(df.x[i]+1, df.y[i]+1, str(i)+"("+str(df.demand[i])+")")
    plt.xlim([-10, 110])
    plt.ylim([-10, 110])
    
    #全ての要素において1であるか確認して図示
    for k in range(vehicle_count):
        for i in range(customer_count):
            for j in range(customer_count):
                if i != j and x[k][i][j] == 1:
                    plt.plot([df.x[i], df.x[j]], [df.y[i], df.y[j]], c="black")
    
    # 図示
    plt.show()
    plt.close()

#解の摂動関数
@jit
def perturbation(x):
    for k in range(vehicle_count):
        if k < (vehicle_count / 2):
            x[k] = 0
    update(x)
    return x

@jit
def func(x): #目的関数
    valu = 0
    for k in range(vehicle_count):    
        valu = valu + np.sum(cost * x[k])
    return valu

@jit
# 解の更新関数
def kmeans_transition(z):
    v = z.copy()
    d = np.zeros((len(z),2))
    for k in range(vehicle_count):
        for m in range(len(z)):
            p = m
            while p == m:
                p = np.random.randint(len(z)) #ランダム選択
            r = np.random.rand() 
            d[m][0] = np.abs(r * (np.sum(z[m][k]) - np.sum(z[p][k]))) #ルートの顧客数
            d[m][1] = np.abs(r * (np.sum(cost *z[m][k])  - np.sum(cost *z[p][k]) )) #ルートの移動距離
                
        #クラスター分析
        model.fit(d);
        ans = model.labels_
        center = []
        for c in range(n):
            cluster_list = d[np.where(ans == c)[0], :] #クラスターリスト作成
            center.append(np.mean(cluster_list)) #クラスター中心算出
            
        center_sorted = np.argsort(center)  #クラスタ中心のソート
        
        #更新
        for c in range(n):
            ID = center_sorted[c] #dの小さいクラスタのインデックス取得
        
            if c + 1 > int(n / 3): #3分の1で確率変更
                P = 0.5
            else:
                P = 0.1
            r = np.random.rand()
            if P > r:
                v[np.where(ans == ID)[0], :] = 0 #確率でリセット
                
    for m in range(len(z)):
        make_route(v[m]) #ルートの補填
    
    return v


# 2値化版ABC
def ABC():
        
    #定数の初期化
    N = customer_count * 2 # 個体数
    tc = np.zeros(N) #カウント
    Cycle = 10 # 周期
    lim = Cycle / 2 #scout bee limit
    z = np.zeros((N,vehicle_count,customer_count,customer_count)) 
    
    #個体の初期化
    for i in range(N):
        z[i] = generate()
    z_best = z[0] #bestの初期化
    best = func(z_best) #bestの値設定
    best_value = []
    
    #ABC　アルゴリズム
    for g in range(Cycle):
        
        #最良値更新
        best_value.append(best)
            
        # employee bee step    
        v = kmeans_transition(z)
        v2 = [] # 追従蜂で選出個体リスト
        i2 = [] # 追従蜂で選出個体インデックスリスト
        
        for i in range(N):
            if func(z[i]) > func(v[i]):
                z[i] = v[i]
                tc[i] = 0
            else: tc[i] += 1
    
        # onlooker bee step
        for i in range(int(N)):
            w = []
            for j in range(N):
                w.append(np.exp(-func(z[j])))
            #追従する個体の選別と記録
            k = roulette_choice(w)
            v2.append(z[k])
            i2.append(k)
            
        v = kmeans_transition(np.array(v2))
        
        for i in i2:
            i3 = int(i2[i])
            if func(z[i3]) > func(v[i3]): 
                z[i3] = v[i3]
                tc[i3] = 0
            else: tc[i3] += 1
    
        # scout bee step
        for i in range(N):
            if tc[i] > lim:
                z[i] = generate()
                tc[i] = 0
    
        # 最良個体の発見
        for i in range(N):
            if best > func(z[i]):
                z_best = z[i]
                best = func(z_best)
    
    return z_best


#メインプログラム

#アルゴリズムの明示
print("KTABC")  
# 解の初期化
x = generate()
plot(x)
C = func(x) #Cost 最小化対象
SC = 0 #Stopping criteria
PI = customer_count #Pertubation Iteration
I = 0 # Iteration
SI = 0 # 無限ループ回避
print('***最初コストは{0}です'.format(np.round(C,2))) 
start = time.time() #時間計測
while customer_count + vehicle_count > SC: #終了判定 (customer_count + vehicle_count 内で更新がなければ終了)
    I = I + 1 #iteration 増加
    #解の摂動
    if I > PI:
        I = 0 #iteration リセット
        SI = SI + 1 # 無限ループ回避カウント1増加
        #print('***SI{0}コストは{1}になりました'.format(SI,np.round(C,2)))
        if SI > 5: #無限ループ回避
            break
        #　解の摂動実行
        x2 = perturbation(x)
        if C > func(x2): #新しい解と目的関数を比較
            # 解の更新
            C = func(x2)
            x = x2
            SC = 0
        SC += 1 #更新しないので１を足す
    else:
        new_solution = ABC() #新しい解の提案
        if C > func(new_solution): #新しい解と目的関数を比較
                # 解の更新
                C = func(new_solution)
                x = new_solution
                SC = 0 #更新されたので0にリセット
        else:
            SC += 1 #更新しないので１を足す
    
elapsed_time = time.time() - start #時間結果
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
print('***最終コストは{0}です'.format(np.round(C,2)))
print('***車両数は{0}です'.format(vehicle_count))
plot(x)
    