# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 15:55:40 2021

@author: kondou_lab

このプログラムは、連続最適化手法のPSOをナップサック問題に適用したプログラムです。
2値化の処理に加えて条件最適化問題を解く手法にアレンジしています。
"""

#ライブラリのインポート
import numpy as np
from numba import jit
import time
from sklearn.cluster import KMeans
import random
import warnings

#jitのwarningの排除
warnings.filterwarnings('ignore')

# 詰める荷物の数
n = 100
# ナップザックの容量
b = 1000
#　ランダム起点固定
np.random.seed(seed=1) 
# クラスターの数
cluster = 5
# 個体数
number_of_particles = 20
# 解リスト
positions = np.zeros((number_of_particles,n))

#kmeansのオブジェクトインスタンス作成
model = KMeans(n_clusters = cluster)

#問題は固定(Value, Size)
v = [29, 1, 2, 7, 22, 28, 17, 4, 24, 4, 80, 21, 87, 1, 14, 5, 77, 38, 60, 4, 94, 20, 5, 57, 7, 4, 6, 14, 22, 43, 5, 42, 4, 43, 79, 48, 12, 12, 63, 63, 73, 44, 21, 38, 48, 12, 27, 15, 73, 3, 17, 59, 76, 6, 68, 4, 84, 7, 73, 31, 1, 1, 65, 45, 4, 55, 17, 44, 52, 76, 21, 1, 1, 15, 17, 10, 24, 17, 11, 97, 12, 53, 2, 4, 40, 28, 30, 42, 33, 3, 28, 32, 20, 11, 45, 5, 62, 94, 1, 57]
s = [25, 2, 8, 12, 23, 25, 15, 5, 24, 9, 45, 20, 49, 5, 13, 8, 51, 37, 39, 5, 54, 18, 8, 47, 16, 5, 12, 25, 30, 33, 8, 37, 6, 36, 51, 46, 12, 21, 49, 42, 48, 40, 18, 27, 37, 11, 26, 22, 49, 4, 19, 47, 49, 6, 40, 10, 51, 8, 43, 37, 2, 3, 46, 32, 8, 35, 29, 42, 43, 47, 29, 4, 3, 15, 29, 9, 30, 17, 13, 53, 15, 36, 8, 6, 34, 21, 31, 32, 35, 6, 33, 38, 18, 13, 31, 14, 41, 51, 2, 36]
initial_position = [(0, 1), (0, 13), (0, 20), (0, 21), (0, 24), (0, 25), (0, 29), (0, 32), (0, 37), (0, 39), (0, 40), (0, 41), (0, 43), (0, 46), (0, 68), (0, 70), (0, 73), (0, 76), (0, 78), (0, 79), (0, 80), (0, 82), (0, 85), (0, 87), (0, 91), (0, 96), (1, 2), (1, 4), (1, 7), (1, 9), (1, 12), (1, 15), (1, 16), (1, 17), (1, 18), (1, 24), (1, 27), (1, 31), (1, 34), (1, 36), (1, 38), (1, 39), (1, 47), (1, 51), (1, 55), (1, 58), (1, 59), (1, 63), (1, 71), (1, 75), (1, 82), (1, 85), (1, 89), (1, 92), (1, 93), (1, 94), (1, 95), (1, 99), (2, 0), (2, 2), (2, 8), (2, 9), (2, 11), (2, 17), (2, 18), (2, 23), (2, 25), (2, 27), (2, 28), (2, 29), (2, 32), (2, 39), (2, 41), (2, 44), (2, 49), (2, 51), (2, 53), (2, 55), (2, 60), (2, 61), (2, 62), (2, 64), (2, 74), (2, 79), (2, 80), (2, 81), (2, 85), (2, 87), (2, 89), (2, 91), (2, 98), (3, 0), (3, 1), (3, 10), (3, 11), (3, 16), (3, 19), (3, 31), (3, 34), (3, 37), (3, 38), (3, 40), (3, 41), (3, 42), (3, 43), (3, 46), (3, 48), (3, 50), (3, 51), (3, 56), (3, 58), (3, 59), (3, 62), (3, 64), (3, 66), (3, 73), (3, 76), (3, 82), (3, 86), (3, 91), (3, 95), (3, 97), (4, 0), (4, 1), (4, 3), (4, 5), (4, 12), (4, 14), (4, 16), (4, 18), (4, 27), (4, 29), (4, 34), (4, 36), (4, 38), (4, 44), (4, 45), (4, 46), (4, 47), (4, 52), (4, 53), (4, 54), (4, 55), (4, 56), (4, 61), (4, 64), (4, 66), (4, 70), (4, 74), (4, 76), (4, 77), (4, 81), (4, 91), (4, 95), (4, 96), (4, 97), (4, 98), (5, 6), (5, 8), (5, 19), (5, 20), (5, 22), (5, 23), (5, 30), (5, 32), (5, 33), (5, 34), (5, 41), (5, 46), (5, 48), (5, 52), (5, 58), (5, 62), (5, 67), (5, 71), (5, 72), (5, 80), (5, 82), (5, 84), (5, 86), (5, 89), (5, 91), (5, 92), (6, 1), (6, 5), (6, 6), (6, 9), (6, 11), (6, 12), (6, 15), (6, 21), (6, 24), (6, 27), (6, 31), (6, 32), (6, 34), (6, 36), (6, 41), (6, 52), (6, 54), (6, 58), (6, 61), (6, 65), (6, 66), (6, 68), (6, 70), (6, 73), (6, 74), (6, 75), (6, 78), (6, 79), (6, 80), (6, 82), (6, 83), (6, 85), (6, 86), (6, 91), (6, 92), (6, 97), (7, 6), (7, 14), (7, 16), (7, 21), (7, 23), (7, 26), (7, 29), (7, 32), (7, 34), (7, 36), (7, 38), (7, 40), (7, 43), (7, 45), (7, 46), (7, 49), (7, 52), (7, 53), (7, 55), (7, 61), (7, 66), (7, 68), (7, 70), (7, 72), (7, 81), (7, 83), (7, 86), (7, 89), (7, 95), (7, 99), (8, 0), (8, 3), (8, 5), (8, 12), (8, 14), (8, 18), (8, 29), (8, 35), (8, 43), (8, 44), (8, 50), (8, 55), (8, 56), (8, 64), (8, 65), (8, 67), (8, 69), (8, 78), (8, 82), (8, 84), (8, 94), (8, 96), (8, 98), (9, 1), (9, 3), (9, 6), (9, 13), (9, 14), (9, 16), (9, 17), (9, 23), (9, 26), (9, 35), (9, 39), (9, 43), (9, 45), (9, 60), (9, 61), (9, 63), (9, 68), (9, 71), (9, 73), (9, 82), (9, 88), (9, 89), (9, 90), (9, 95), (9, 99), (10, 1), (10, 2), (10, 3), (10, 12), (10, 23), (10, 26), (10, 28), (10, 29), (10, 34), (10, 38), (10, 39), (10, 43), (10, 44), (10, 46), (10, 48), (10, 51), (10, 54), (10, 55), (10, 57), (10, 60), (10, 63), (10, 68), (10, 71), (10, 73), (10, 74), (10, 79), (10, 81), (10, 89), (10, 91), (10, 97), (10, 98), (11, 1), (11, 4), (11, 6), (11, 8), (11, 9), (11, 12), (11, 18), (11, 22), (11, 25), (11, 36), (11, 37), (11, 38), (11, 39), (11, 45), (11, 50), (11, 51), (11, 52), (11, 57), (11, 59), (11, 65), (11, 74), (11, 75), (11, 77), (11, 79), (11, 84), (11, 86), (11, 87), (11, 90), (11, 92), (11, 94), (11, 99), (12, 0), (12, 2), (12, 7), (12, 9), (12, 17), (12, 18), (12, 23), (12, 27), (12, 29), (12, 33), (12, 36), (12, 43), (12, 44), (12, 45), (12, 47), (12, 51), (12, 55), (12, 58), (12, 61), (12, 62), (12, 69), (12, 71), (12, 80), (12, 89), (12, 90), (12, 91), (12, 93), (12, 95), (12, 98), (13, 3), (13, 6), (13, 15), (13, 18), (13, 20), (13, 21), (13, 22), (13, 26), (13, 45), (13, 47), (13, 49), (13, 51), (13, 56), (13, 59), (13, 61), (13, 64), (13, 65), (13, 67), (13, 68), (13, 69), (13, 70), (13, 75), (13, 77), (13, 83), (13, 84), (13, 89), (13, 90), (13, 92), (13, 95), (13, 96), (13, 98), (13, 99), (14, 2), (14, 5), (14, 6), (14, 12), (14, 13), (14, 15), (14, 16), (14, 17), (14, 25), (14, 31), (14, 32), (14, 33), (14, 34), (14, 39), (14, 41), (14, 42), (14, 43), (14, 45), (14, 56), (14, 61), (14, 62), (14, 76), (14, 82), (14, 83), (14, 84), (14, 86), (14, 87), (14, 94), (14, 96), (14, 99), (15, 7), (15, 12), (15, 14), (15, 18), (15, 19), (15, 21), (15, 22), (15, 33), (15, 39), (15, 45), (15, 46), (15, 49), (15, 50), (15, 51), (15, 53), (15, 54), (15, 61), (15, 63), (15, 66), (15, 67), (15, 70), (15, 75), (15, 86), (15, 87), (15, 88), (15, 89), (15, 91), (15, 96), (15, 98), (15, 99), (16, 3), (16, 5), (16, 7), (16, 10), (16, 20), (16, 21), (16, 25), (16, 27), (16, 34), (16, 37), (16, 39), (16, 43), (16, 44), (16, 45), (16, 53), (16, 54), (16, 56), (16, 63), (16, 64), (16, 67), (16, 68), (16, 69), (16, 71), (16, 74), (16, 75), (16, 77), (16, 78), (16, 85), (16, 88), (16, 89), (16, 91), (16, 93), (16, 95), (16, 96), (16, 97), (16, 99), (17, 1), (17, 3), (17, 5), (17, 12), (17, 15), (17, 17), (17, 20), (17, 22), (17, 23), (17, 28), (17, 31), (17, 32), (17, 34), (17, 36), (17, 40), (17, 42), (17, 44), (17, 46), (17, 47), (17, 49), (17, 55), (17, 61), (17, 69), (17, 72), (17, 74), (17, 75), (17, 80), (17, 81), (17, 84), (17, 85), (17, 87), (17, 89), (17, 90), (17, 92), (17, 93), (17, 94), (17, 97), (17, 98), (18, 0), (18, 3), (18, 4), (18, 5), (18, 10), (18, 11), (18, 19), (18, 22), (18, 24), (18, 29), (18, 32), (18, 39), (18, 42), (18, 43), (18, 47), (18, 48), (18, 51), (18, 53), (18, 54), (18, 58), (18, 63), (18, 70), (18, 78), (18, 80), (18, 83), (18, 85), (18, 88), (18, 90), (18, 91), (18, 95), (18, 97), (18, 99), (19, 2), (19, 13), (19, 19), (19, 20), (19, 21), (19, 22), (19, 26), (19, 28), (19, 29), (19, 30), (19, 34), (19, 36), (19, 39), (19, 49), (19, 51), (19, 52), (19, 60), (19, 61), (19, 62), (19, 63), (19, 65), (19, 78), (19, 82), (19, 83), (19, 86), (19, 90), (19, 93), (19, 94), (19, 97), (19, 99)]

T = []
V = []
list_ans = []

optimal = 1609.0 #最適値

@jit
#xに０または1を入れて初期化
def generate():
    x = np.zeros(n) # 答えの初期化
    for i in range(n):
        r = np.random.random()
        if r > 0.7:
            x[i] = 1
    return x

@jit
def func(x): #目的関数
    valu = 0    
    valu = valu + np.sum(v * x)
    return valu

@jit
# 摂動関数
def perturbation(x):
    chenge_rate = 0.05 # 変動率
    for i in range(n):
        r = np.random.random()
        if r < chenge_rate:
            if x[i] == 0:
                x[i] = 1
            else:
                x[i] = 0
            
    return x

@jit
# 解の更新関数
def kmeans_transition(positions, velocities, best_position):
    
    y = positions.copy()
    
    for i in range(len(positions)):
        d = np.zeros(n)
            
        for m in range(n):
            # 速度を元にdを定義
            d[m] = velocities[i][m]  #荷物の値
        
        #クラスター分析 (自作関数では精度なし)
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
                            break
                    
                    # 更新
                    y[i][tag] = best_position[tag]
    
    return y

        
@jit
# PSOの枠組み
def PSO():
    # 定数初期化
    number_of_particles = 20
    limit_times = 500
    positions = np.zeros((number_of_particles,n))
    w=0.5
    ro_max=0.14
    ans_value = []
    best = 0
    PI = 0

    # 各粒子の位置
    for i,j in initial_position:
        positions[i][j] = 1
    
    velocities = np.zeros(positions.shape)
    
    # 各粒子ごとのパーソナルベスト位置
    personal_best_positions = np.copy(positions)
    
    personal_best_scores = np.zeros(number_of_particles)
    
    # 各粒子ごとのパーソナルベストの値
    for m in range(number_of_particles):
        personal_best_scores[m] = func(personal_best_positions[m])
    
    # グローバルベスト位置
    global_best_particle_position = np.zeros(n)
    
    # 規定回数
    for T in range(limit_times):
        
        #進捗報告
        if T % 100 == 0:
            print("今{}回目".format(T))
        
        # 摂動しないと厳しい
        if PI > 50:
            PI = 0
            for i in range(number_of_particles):
                positions[i] = perturbation(positions[i])
        
        PI += 1
        
        # 速度更新
        for i in range(number_of_particles):
            for j in range(n):
                rc1 = random.uniform(0, ro_max)
                rc2 = random.uniform(0, ro_max)
                
                # 速度の更新（PSOと同様）
                velocities[i][j] = np.abs(velocities[i][j] * w + rc1 * (personal_best_positions[i][j] - positions[i][j]) + rc2 * (
                    global_best_particle_position[j] - positions[i][j]))
        
        # 位置更新
        positions = kmeans_transition(positions, velocities,global_best_particle_position)
    
        # パーソナルベストの更新
        for m in range(number_of_particles):
            score = func(positions[m])
            
            # 違反度計算
            vio_posi = np.sum(s * positions[m]) - b
            vio_best = np.sum(s * personal_best_positions[m]) - b
            
            # 違反度によって分岐
            if vio_best > 0 and vio_posi < 0:
                personal_best_scores[m] = score
                personal_best_positions[m] = positions[m]
                PI = 0
                
            if vio_best < 0 and vio_posi < 0:
                if score > personal_best_scores[m]:
                    personal_best_scores[m] = score
                    personal_best_positions[m] = positions[m]
                    PI = 0
                    
            if vio_best > 0 and vio_posi > 0:
                if (vio_best - vio_posi) > 0:
                    personal_best_scores[m] = score
                    personal_best_positions[m] = positions[m]
                    PI = 0
                    
        # グローバルベストの更新
        for i in range(number_of_particles):
            if best < personal_best_scores[i]:
                if np.sum(s * personal_best_positions[i]) < b:
                    global_best_particle_position = personal_best_positions[i].copy()
                    
        if best < func(global_best_particle_position):
            best = func(global_best_particle_position)
            ans_value.append((T,best))
        
    return global_best_particle_position,ans_value


#メインプログラム

# リストの初期化
ans = [] # 答えの番号
x = np.zeros(n)
ans_value = [] #答えの目的関数
start = time.time() #時間計測
x,ans_value = PSO() #PSO
C = func(x)


# 結果
print("Result:")
for i in range(n):
    #print(a[i].getName(), int(a[i].value()))
    if x[i] == 1:
        ans.append(i)

#出力
print("番号")
print(ans)
list_ans.append(ans_value)
elapsed_time = time.time() - start #時間結果
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
print('***最終バリューは{0}です'.format(np.round(C,2)))
print('***最終重量は{0}です'.format(np.round(np.sum(s * x))))
print('***最適値は{0}です'.format(optimal))
T.append(elapsed_time)
V.append(np.round(C,2))