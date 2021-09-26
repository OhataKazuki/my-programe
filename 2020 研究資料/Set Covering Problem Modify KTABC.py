# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:45:18 2020

@author: Kazuki Ohata

このプログラムは、連続最適化手法のABCをSCPに適用したプログラムです。
2値化の処理に加えて条件最適化問題を解く手法にアレンジしています。
"""

#ライブラリのインポート
import numpy as np
from numba import jit
import time
import random
from sklearn.cluster import KMeans
import warnings

#jitのwarningの排除
warnings.filterwarnings('ignore')

# 満たすべき行の規模
n = 20
#列の数
col = 100
#　ランダム起点固定
np.random.seed(seed=1) 
# クラスターの数
cluster = 5

#kmeansのオブジェクトインスタンス作成
model = KMeans(n_clusters = cluster)

# 実験用のリスト
V = []
T = []
list_ans = []

#問題の定義
A = np.zeros((n,col))
cost = np.zeros(col)

#問題は固定
position = [(0, 12), (0, 24), (0, 29), (0, 30), (0, 38), (0, 40), (0, 46), (0, 65), (0, 85), (0, 87), (0, 91), (1, 4), (1, 14), (1, 21), (1, 22), (1, 25), (1, 40), (1, 58), (1, 63), (1, 76), (1, 91), (1, 95), (2, 2), (2, 14), (2, 26), (2, 50), (2, 58), (2, 60), (2, 61), (2, 72), (2, 73), (2, 75), (2, 77), (2, 89), (2, 99), (3, 2), (3, 8), (3, 17), (3, 25), (3, 34), (3, 36), (3, 49), (3, 57), (3, 62), (3, 65), (4, 0), (4, 5), (4, 22), (4, 26), (4, 29), (4, 31), (4, 32), (4, 42), (4, 68), (4, 71), (4, 75), (4, 79), (4, 97), (4, 98), (5, 3), (5, 8), (5, 61), (5, 70), (5, 73), (5, 83), (5, 89), (5, 94), (6, 5), (6, 13), (6, 16), (6, 17), (6, 20), (6, 22), (6, 31), (6, 59), (6, 62), (6, 77), (6, 87), (7, 4), (7, 7), (7, 11), (7, 18), (7, 24), (7, 45), (7, 56), (7, 61), (7, 65), (7, 71), (7, 77), (7, 93), (7, 94), (8, 72), (8, 85), (9, 9), (9, 15), (9, 16), (9, 28), (9, 31), (9, 44), (9, 61), (9, 70), (9, 75), (9, 93), (10, 15), (10, 19), (10, 34), (10, 36), (10, 53), (10, 55), (10, 68), (10, 73), (10, 94), (11, 0), (11, 6), (11, 26), (11, 29), (11, 36), (11, 37), (11, 46), (11, 49), (11, 94), (12, 14), (12, 21), (12, 43), (12, 44), (12, 58), (12, 61), (12, 62), (12, 66), (12, 69), (12, 71), (12, 86), (12, 94), (13, 6), (13, 16), (13, 17), (13, 26), (13, 46), (13, 88), (13, 96), (13, 98), (14, 7), (14, 10), (14, 14), (14, 16), (14, 23), (14, 34), (14, 39), (14, 47), (14, 96), (15, 4), (15, 24), (15, 32), (15, 40), (15, 58), (15, 60), (15, 65), (15, 74), (15, 90), (16, 0), (16, 1), (16, 5), (16, 25), (16, 26), (16, 27), (16, 46), (16, 47), (16, 48), (16, 51), (16, 52), (16, 81), (16, 86), (16, 92), (17, 22), (17, 31), (17, 47), (17, 75), (17, 78), (17, 84), (17, 93), (18, 2), (18, 5), (18, 34), (18, 45), (18, 52), (18, 54), (18, 57), (18, 63), (18, 74), (18, 79), (19, 33), (19, 38), (19, 41), (19, 43), (19, 63), (19, 67), (19, 77), (19, 88)]
cost = [15.0463, 10.308, 16.2863, 11.2086, 14.7791, 14.5227, 13.8796, 12.2538, 13.2089, 11.6483, 11.7833, 14.2512, 11.1784, 9.2082, 15.1693, 14.9105, 15.8114, 15.4796, 12.5489, 13.8381, 11.7686, 13.7811, 16.817, 9.9709, 14.0416, 15.7535, 17.2235, 11.6251, 12.5882, 14.8893, 11.3497, 15.904, 13.5648, 14.0638, 15.3779, 8.7971, 14.8427, 11.6915, 13.1838, 13.2554, 14.1042, 12.0303, 12.8323, 13.7427, 11.669, 12.6367, 16.71, 14.9709, 12.0443, 14.8686, 12.9321, 10.805, 14.4599, 13.8618, 13.483, 11.5429, 12.3256, 15.3536, 18.8251, 13.0282, 13.6119, 18.0789, 16.3764, 15.8652, 11.2762, 16.6794, 11.5338, 12.0891, 14.246, 14.9516, 14.3068, 15.0015, 14.0961, 12.3494, 15.6721, 15.3188, 10.7025, 15.853, 12.2276, 12.1011, 12.5462, 13.2642, 9.2546, 13.8166, 13.4622, 14.0959, 13.3354, 13.8278, 12.8876, 13.8901, 10.1278, 12.1725, 12.2982, 14.8516, 19.0172, 10.6323, 13.6237, 10.4519, 11.807, 12.3063]

for i,j in position:
    A[i][j] = 1

optimal = 101


@jit
# 確率計算関数
def calculate_probabilities(N,z):
    # probabilities
    p = np.zeros(N)
    # fitness
    f = np.zeros(N)
    # violation
    vio = np.zeros(N)
    # 各リストの作成
    for i in range(N):
        f[i] = func(z[i])
        vio[i] = np.sum(np.where(np.dot(A,z[i]) > 0 ,0,1))
    
    # 確率計算
    for i in range(N):
        # 条件を満たしているとき
        if vio[i] == 0:
            p[i] = 0.5 + (f[i] / np.sum(f) * 0.5)
        # 条件を満たしていないとき
        else:
            p[i] = (1 - (vio[i] / np.sum(vio))) * 0.5
    
    return p

# 条件修正関数（jitだとうまくいかない）
def repair(x):
    x_repaired = x.copy()
    check = 1
    # 条件満たさなかったらヒューリスティクスで補強
    while check == 1:
        check = 0
        x_repaired[heuristics(x_repaired)] = 1
        if np.dot(A,x_repaired).all() < 1:
            check = 1
    return x_repaired


@jit
# 解の初期化
def initialisation():
    # xの宣言
    x = np.zeros(col)
    for i in range(col):
        # 2分の1で振り分け
        r = random.random()
        if r < 0.5:
            x[i] = 1
        else:
            x[i] = 0
    
    if np.dot(A,x).all() < 1:  
        x = repair(x)
    
    return x


# 解の補強
def heuristics(x):
    # coverされていない行のリスト
    R = []
    L = []
    for i in range(n):
        # coverされていない行の割り出し
        if np.dot(A,x)[i] < 1:
            R.append(i)
            
    for i in R:
        for j in range(col):
            if A[i][j] == 1:
                L.append(j)
                
    index = np.random.choice(L)
    
    return index


#解の摂動
@jit
def perturbation(x):
    # 変更する数を算出
    num_chenge = int(col * 0.05)
    for i in range(num_chenge):
        # ループをブレイクで脱出
        for j in range(col):
            index = int(col * random.random())
            if x[index] == 1:
                x[index] = 0
                break
    
    return x

@jit
# 評価関数
def func(x):
    # costの合計を算出
    value = np.sum(cost * x)
    return value


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
# 解の更新関数
def kmeans_transition(i,z):
    y = z[i].copy()
    d = np.zeros(col)
        
    for m in range(col):
        # 速度を元にdを定義
        p = m
        while p == m:
            p = np.random.randint(len(z)) #ランダム選択
        r = (np.random.rand() * 2) - 1
        d[m] = np.abs(r * (z[i][m] - z[p][m])) #荷物の値
        
    #クラスター分析 
    model.fit(d.reshape((-1,1)));
    ans = model.labels_
    
    center = []
    
    for c in range(cluster):
        cluster_list = d[np.where(ans == c)] #クラスターリスト作成
        center.append(np.mean(cluster_list)) #クラスター中心算出
    
    center_sorted = np.argsort(center)  #クラスタ中心のソート
    
    
    #更新
    for c in range(cluster):
        ID = center_sorted[c] #dの小さいクラスタのインデックス取得
        
        if c + 1 > int(cluster / 3): #3分の1で確率変更
            P = 0.5
        else:
            P = 0.1
        
        # クラスターにある個体の数だけ繰り返す
        for j in d[np.where(ans == ID)]:
            r = np.random.rand()

            if P > r:
                # 変更箇所のタグ付け
                for m in range(n):
                    if d[m] == j: 
                        tag = m
                
                        # 更新
                        if z[i][tag] == 0:
                            y[tag] = 1#確率でリセット
                        else:
                            y[tag] = 0 
    
    return y
        

@jit
# 2値化版ABC
def ABC():
        
    #定数の初期化
    N = 10# 個体数
    tc = np.zeros(N) #カウント
    Cycle = 500 # 周期
    lim = Cycle / 2
    z = np.zeros((N,col))
    for i in range(N):
        z[i] = initialisation()
    z_best = z[0] #x_bestの初期化
    best = func(z_best)
    best_value = []
    ans_value = []
    PI = 0 # perturbetion Iteration
    
    #ABC　アルゴリズム
    for g in range(Cycle):
        
        #進捗報告
        if g % 100 == 0:
            print("今{}回目".format(g))
        
        best_value.append(best)
        
        if PI > 50:
            PI = 0
            for i in range(N):
                z[i] = perturbation(z[i])
        
        PI += 1
        
        # employee bee step
        for i in range(N):
            v = kmeans_transition(i,z)

            # 違反度計算
            vio_z = np.sum(np.where(np.dot(A,z[i]) > 0 ,0,1))
            vio_v = np.sum(np.where(np.dot(A,v) > 0 ,0,1))
            
            # 違反度によって分岐
            if vio_z > 0 and vio_v <= 0:
                    z[i] = v
                    tc[i] = 0
                    PI = 0
            else: tc[i] += 1 
            
            if vio_z <= 0 and vio_v <= 0:
                if func(z[i]) > func(v):
                    z[i] = v
                    tc[i] = 0
                    PI = 0
            else: tc[i] += 1      
                    
            if vio_z > 0 and vio_v > 0:
                if (vio_z - vio_v) > 0:
                    z[i] = v
                    tc[i] = 0
                    PI = 0
            else: tc[i] += 1
            
        # 各個体の確率計算
        p = calculate_probabilities(N,z)
        
        # onlooker bee step
        for i in range(N):
            # 乱数生成
            r = np.random.random()
            
            if r < p[i]:
                v = kmeans_transition(i,z)
                
                # 違反度計算
                vio_z = np.sum(np.where(np.dot(A,z[i]) > 0 ,0,1))
                vio_v = np.sum(np.where(np.dot(A,v) > 0 ,0,1))
                
                # 違反度によって分岐
                if vio_z > 0 and vio_v <= 0:
                        z[i] = v
                        tc[i] = 0
                        PI = 0
                else: tc[i] += 1 
                
                if vio_z <= 0 and vio_v <= 0:
                    if func(z[i]) > func(v):
                        z[i] = v
                        tc[i] = 0
                        PI = 0
                else: tc[i] += 1
                
                if vio_z > 0 and vio_v > 0:
                    if (vio_z - vio_v) > 0:
                        z[i] = v
                        tc[i] = 0
                        PI = 0
                else: tc[i] += 1
        
        # scout bee step
        for i in range(N):
            if tc[i] > lim:
                z[i] = initialisation()
                tc[i] = 0   
        
        # 最良個体の発見
        for i in range(N):
            if best > func(z[i]):
                z_best = z[i].copy()
                best = func(z_best)
                ans_value.append((g,np.round(best,2)))
    
    return z_best,ans_value


#メインプログラム

# リストの初期化
ans = [] # 答えの番号
x = np.zeros(col)
ans_value = []
start = time.time() #時間計測
x,ans_value = ABC() #新しい解の提案
Value = func(x)


# 結果
print("Result:")
for i in range(col):
    #print(a[i].getName(), int(a[i].value()))
    if x[i] == 1:
        ans.append(i)

#出力
print(ans)
list_ans.append(ans_value)
elapsed_time = time.time() - start #時間結果
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
print('***最終バリューは{0}です'.format(np.round(Value,2)))
print("最適値は、{}です".format(optimal))
T.append(elapsed_time)
V.append(np.round(Value,2))

