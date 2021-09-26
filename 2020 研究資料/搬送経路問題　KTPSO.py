# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:16:24 2020

@author: Kazuki Ohata

このプログラムは、連続最適化のPSOで搬送経路最適化問題を解くプログラムです。
離散値を確率に写像させて01を反転させる(kmeans-transition)
"""

# ライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from scipy.spatial import distance_matrix
import itertools
from sklearn.cluster import KMeans
import time
from numba import jit
import warnings

#warningを無くす
warnings.filterwarnings('ignore') 

customer_count = 12 #顧客数（id=0はdepot）
vehicle_capacity = 40 #車両容量

#乱数の初期値固定
np.random.seed(seed=3) 
n = 10 #　クラスターの数

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

#車両数決定
vehicle_count = int(demand_sum / vehicle_capacity) + 1

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
                    #print("go to {0} from {1}".format(j,i)) #デバッグ用
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
        #print("go to 0 from {0}".format(i)) #デバッグ用
        #print(i) #デバッグ用
        #plot(x)
    # 各ルート最適化
    initialize(x)
    
    return x

@jit
# 引数ありのgenerate関数
def make_route(v):
    check = list(range(1,customer_count))
    
    #check確認
    for k in range(vehicle_count):
        for i in range(customer_count): 
            for j in range(customer_count):
                if v[k][i][j] == 1:
                    if j != 0:
                        check.remove(j)
                    #print(j) #デバッグ用
    
    for k in range(vehicle_count):   
        capacity = 0
        i = 0 #iを初期化
        while capacity < vehicle_capacity:
            if len(check) > 0:
                    # ランダムで選択
                r = int(np.random.random() * len(check))
                j = check[r]
                if capacity + df.demand[j] < vehicle_capacity:
                    #print("go to {0} from {1}".format(j,i)) #デバッグ用
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
        #print(i) #デバッグ用
        #plot(x)
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
        #print(i) #デバッグ用
        #plot(x)
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
    
    #地図の図示
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

#目的関数
@jit
def func(x): 
    valu = 0
    for k in range(vehicle_count):    
        valu = valu + np.sum(cost * x[k])
    return valu

# 各粒子の速度更新(PSOの一部)
@jit
def update_velocities(positions,velocities, personal_best_positions, global_best_particle_position, w=0.5,
                      ro_max=0.14):
    rc1 = random.uniform(0, ro_max)
    rc2 = random.uniform(0, ro_max)

    # 速度の更新（PSOと同様）
    velocities = np.abs(velocities * w + rc1 * (personal_best_positions - positions) + rc2 * (
            global_best_particle_position - positions))
    return velocities


# 解の更新関数
@jit
def kmeans_transition(positions,velocities):
    v = positions.copy()
    d = np.zeros((len(positions),2))
    
    for k in range(vehicle_count):
        for m in range(len(positions)):
            # 速度を元にdを定義
            d[m][0] = np.sum(velocities[m][k]) #速度の総和
            d[m][1] = np.sum(cost * velocities[m][k]) # 速度のコスト
        
        #クラスター分析
        model.fit(d);
        ans = model.labels_ #クラスターのラベルの取得
        center = []
        for c in range(n):
            cluster_list = d[np.where(ans == c)[0], :] #クラスターリスト作成
            center.append(np.mean(cluster_list)) #クラスター中心算出
            
        center_sorted = np.argsort(center) #クラスタ中心のソート
        #更新
        for c in range(n):
            ID = center_sorted[c]  #dの小さいクラスタのインデックス取得
        
            if c + 1 > int(n / 3): #3分の1で確率変更
                P = 0.5
            else:
                P = 0.1
            r = np.random.rand()
            if P > r:
                v[np.where(ans == ID)[0], :] = 0 #確率でリセット
                
    for m in range(len(positions)):
        make_route(v[m]) #ルートの補填
    
    return v

# PSOの枠組み
@jit
def PSO():
    # 定数初期化
    number_of_particles = customer_count * 2
    limit_times = 10
    positions = np.zeros((number_of_particles,vehicle_count,customer_count,customer_count))

    
    # 各粒子の位置
    for m in range(number_of_particles):
        positions[m] = generate()
    
    velocities = np.zeros(positions.shape)
    
    # 各粒子ごとのパーソナルベスト位置
    personal_best_positions = np.copy(positions)
    
    personal_best_scores = np.zeros(number_of_particles)
    
    # 各粒子ごとのパーソナルベストの値
    for m in range(number_of_particles):
        personal_best_scores[m] = func(personal_best_positions[m])
    
    # グローバルベストの粒子ID
    global_best_particle_id = np.argmin(personal_best_scores)
    
    # グローバルベスト位置
    global_best_particle_position = personal_best_positions[global_best_particle_id]
    
    # 規定回数
    for T in range(limit_times):
        
        # 速度更新
        velocities = update_velocities(positions, velocities, personal_best_positions,
                                       global_best_particle_position)
        
        # 位置更新
        positions = kmeans_transition(positions, velocities)
    
        # パーソナルベストの更新
        for m in range(number_of_particles):
            score = func(positions[m])
            if score < personal_best_scores[m]:
                personal_best_scores[m] = score
                personal_best_positions[m] = positions[m]
    
        # グローバルベストの更新
        global_best_particle_id = np.argmin(personal_best_scores)
        global_best_particle_position = personal_best_positions[global_best_particle_id]
        
    return global_best_particle_position

#メインプログラム

#アルゴリズムの明示
print("KTPSO")
# 解の初期化
x = generate()
#制約条件確認
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
    #解の探索
    else:
        new_solution = PSO() #新しい解の提案
        if C > func(new_solution): #新しい解と目的関数を比較
                # 解の更新
                C = func(new_solution)
                x = new_solution
                SC = 0 #更新されたので0にリセット
        else:
            SC += 1 #更新しないので１を足す


#探索結果の出力
elapsed_time = time.time() - start #時間結果
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]") #経過時間の出力
print('***最終コストは{0}です'.format(np.round(C,2))) #最終探索の出力
print('***車両数は{0}です'.format(vehicle_count)) #車両の使用台数の出力
plot(x) #最良ルートの表示