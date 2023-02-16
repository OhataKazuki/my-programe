# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 00:29:39 2022

@author: kondou_lab
"""

#ロボットの逆運動学問題を人工蜂コロニーアルゴリズムで解くプログラム

#ライブラリのインポート
import numpy as np
import warnings
import math

#seed値の決定
np.random.seed(1)
warnings.simplefilter('ignore')

#パラメータ設定
s = 100
c1 = 1.494
c2 = 1.494
w = 0.729

lim = 80
TC = np.zeros(s) #更新カウント

MaxIT = 500
frequency = 50 #問題変化頻度
Tv = -0.5
xmin = [-180, -90, -90, -90, -90, -90, -30]
xmax = [180, 30, 120, 90, 90, 90, 90]
a = [0, 0.2, 0.25, 0.3, 0.2, 0.2, 0.1]
alpha = [-90, 90, -90, 90, -90, 0, 0]
target = [45, 0, 45, 0, 45, 0, 0]
#target = [100, -30, -56, 17, 62, 38, 21]
d = [0.5, 0, 0, 0, 0, 0, 0.05]
dim = 7

#逆運動学を解く関数
def func(x):

    cos = []
    sin = []
    for i in range(len(x)):
        x_rad = math.radians(x[i])
        cos.append(math.cos(x_rad))
        sin.append(math.sin(x_rad))
    
    Mx = (cos[0] * cos[1] * cos[2] * cos[3]  - sin[0] * sin[2] * sin[3]  - cos[0] * sin[1] * sin[3]) * (cos[4] * cos[5] * a[6] * cos[6] - cos[4] * sin[5] * a[6] * sin[6] - sin[4] * d[6] + cos[4] * a[5] * cos[5] + a[4] * cos[4]) * \
        (-cos[0] * cos[1] * sin[2] - sin[0] * cos[2]) * (sin[4] * cos[5] * a[6] * cos[6] - sin[4] * sin[5] * a[6] * sin[6] + cos[4] * d[6] + sin[4] * cos[5] * a[5] + a[4] * sin[4]) * \
        (cos[0] * cos[1] * cos[2] * sin[3] * sin[0] * sin[2] * sin[3] * cos[0] * cos[3] * sin[1]) * (-sin[5] * a[6] * cos[6] - cos[5] * a[6] * sin[6] - a[5] * sin[5]) + \
        cos[0] * cos[1] * (cos[2] * cos[3] * a[3] + a[2] * cos[2]) - sin[0] * (sin[2] * cos[3] * a[3] + a[2] * sin[2]) - \
        cos[0] * sin[1] * a[3] * sin[3] + cos[0] * cos[1] * a[1]
    
    """
    a1 = (cos[0] * cos[1] * cos[2] * cos[3]  - sin[0] * sin[2] * sin[3]  - cos[0] * sin[1] * sin[3]) * (cos[4] * cos[5] * a[6] * cos[6] - cos[4] * sin[5] * a[6] * sin[6] - sin[4] * d[6] + cos[4] * a[5] * cos[5] + a[4] * cos[4])
    a2 = (-cos[0] * cos[1] * sin[2] - sin[0] * cos[2]) * (sin[4] * cos[5] * a[6] * cos[6] - sin[4] * sin[5] * a[6] * sin[6] + cos[4] * d[6] + sin[4] * cos[5] * a[5] + a[4] * sin[4])
    a3 = (cos[0] * cos[1] * cos[2] * sin[3] * sin[0] * sin[2] * sin[3] * cos[0] * cos[3] * sin[1]) * (-sin[5] * a[6] * cos[6] - cos[5] * a[6] * sin[6] - a[5] * sin[5]) 
    a4 = cos[0] * cos[1] * (cos[2] * cos[3] * a[3] + a[2] * cos[2])   
    a5 = -sin[0] * (sin[2] * cos[3] * a[3] + a[2] * sin[2])
    a6 = -cos[0] * sin[1] * a[3] * sin[3]
    a7 = cos[0] * cos[1] * a[1]
    
    
    print("-------------")
    print(a4)
    print(a5)
    print(a6)
    print(a7)
    
    Mx = a1 * a2 * a3 + a4 + a5 + a6 + a7
    """
    
    My = (sin[0] * cos[1] * cos[2] * cos[3] + cos[0] * sin[2] * cos[3] - sin[0] * sin[1] * sin[3]) * (cos[4] * cos[5] * a[6] * cos[6] - cos[4] * sin[5] * a[6] * sin[6] - sin[4] * d[6] + cos[4] * cos[5] * a[5] + a[4] * cos[4]) + \
        (-sin[0] * cos[1] * sin[2] + cos[0] * cos[2]) * (sin[4] * cos[5] * a[6] * cos[6] - sin[4] * sin[5] * a[6] * sin[6] + cos[4] * d[6] + sin[4] * cos[5] * a[5] + a[4] * sin[4]) + \
        (sin[0] * cos[1] * cos[2] * sin[3] + cos[0] * sin[2] * sin[3] + sin[0] * sin[1] * cos[3]) * (-sin[5] * a[6] * cos[6] - cos[5] * a[6] * sin[6] - a[5] * sin[5]) + \
        sin[0] * cos[1] * (cos[2] * cos[3] * a[3] + a[2] * cos[2]) + cos[0] * (sin[2] * cos[3] * a[3] + a[2] * sin[2]) - \
        sin[0] * sin[1] * sin[3] * a[3] + sin[0] * cos[1] * a[1]
    
    """
    a8 = (sin[0] * cos[1] * cos[2] * cos[3] + cos[0] * sin[2] * cos[3] - sin[0] * sin[1] * sin[3]) * (cos[4] * cos[5] * a[6] * cos[6] - cos[4] * sin[5] * a[6] * sin[6] - sin[4] * d[6] + cos[4] * cos[5] * a[5] + a[4] * cos[4])
    a9 = (-sin[0] * cos[1] * sin[2] + cos[0] * cos[2]) * (sin[4] * cos[5] * a[6] * cos[6] - sin[4] * sin[5] * a[6] * sin[6] + cos[4] * d[6] + sin[4] * cos[5] * a[5] + a[4] * sin[4])
    a10 = (sin[0] * cos[1] * cos[2] * sin[3] + cos[0] * sin[2] * sin[3] + sin[0] * sin[1] * cos[3]) * (-sin[5] * a[6] * cos[6] - cos[5] * a[6] * sin[6] - a[5] * sin[5])
    a11 = sin[0] * cos[1] * (cos[2] * cos[3] * a[3] + a[2] * cos[2]) + cos[0] * (sin[2] * cos[3] * a[3] + a[2] * sin[2]) - sin[0] * sin[1] * sin[3] * a[3] + sin[0] * cos[1] * a[1]
    
    print("////////////////////")
    print(a8)
    print(a9)
    print(a10)
    print(a11)
    
    My = a8 + a9 + a10 + a11
    """
    
    Mz =(-sin[1] * cos[2] * cos[3] - cos[1] * sin[3]) * (cos[4] * cos[5] * a[6] * cos[6] - cos[4] * sin[5] * a[6] * sin[6] - sin[4] * d[6] + cos[4] * cos[5] * a[5] + a[4] * cos[4]) + \
        sin[1] * sin[2] * (sin[4] * cos[5] * a[6] * cos[6] - sin[4] * sin[5] * a[6] * sin[6] + cos[4] * d[6] + sin[4] * cos[5] * a[5] + sin[4] * a[4]) + (-sin[1] * cos[2] * sin[3] + cos[1] * cos[3]) * \
        (-sin[5] * a[6] * cos[6] - cos[5] * a[6] * sin[6] - sin[5] * a[5]) - sin[1] * (cos[2] * cos[3] * a[3] + a[2] * cos[2]) - \
        cos[1] * sin[3] * a[3] - sin[1] * a[1] + d[0]
        
    return np.array([Mx, My, Mz])

#逆運動学問題を解く関数（別表記）
def func2(x):

    cos = []
    sin = []
    for i in range(len(x)):
        x_rad = math.radians(x[i])
        cos.append(math.cos(x_rad))
        sin.append(math.sin(x_rad))
    
    Mx = (cos[0] * cos[1] * cos[2] * cos[3]  - sin[0] * sin[2] * sin[3]  - cos[0] * sin[1] * sin[3]) * (cos[4] * cos[5] * a[6] * cos[6] - cos[4] * sin[5] * a[6] * sin[6] - sin[4] * d[6] + cos[4] * a[5] * cos[5] + a[4] * cos[4]) * \
        (-cos[0] * cos[1] * sin[2] - sin[0] * cos[2]) * (sin[4] * cos[5] * a[6] * cos[6] - sin[4] * sin[5] * a[6] * sin[6] + cos[4] * d[6] + sin[4] * cos[5] * a[5] + a[4] * sin[4]) * \
        (cos[0] * cos[1] * cos[2] * sin[3] * sin[0] * sin[2] * sin[3] * cos[0] * cos[3] * sin[1]) * (-sin[5] * a[6] * cos[6] - cos[5] * a[6] * sin[6] - a[5] * sin[5]) + \
        cos[0] * cos[1] * (cos[2] * cos[3] * a[3] + a[2] * cos[2]) - sin[0] * (sin[2] * cos[3] * a[3] + a[2] * sin[2]) - \
        cos[0] * sin[1] * a[3] * sin[3] + cos[0] * cos[1] * a[1]
    
    My = (sin[0] * cos[1] * cos[2] * cos[3] + cos[0] * sin[2] * cos[3] - sin[0] * sin[1] * sin[3]) * (cos[4] * cos[5] * a[6] * cos[6] - cos[4] * sin[5] * a[6] * sin[6] - sin[4] * d[6] + cos[4] * cos[5] * a[5] + a[4] * cos[4]) + \
        (-sin[0] * cos[1] * sin[2] + cos[0] * cos[2]) * (sin[4] * cos[5] * a[6] * cos[6] - sin[4] * sin[5] * a[6] * sin[6] + cos[4] * d[6] + sin[4] * cos[5] * a[5] + a[4] * sin[4]) + \
        (sin[0] * cos[1] * cos[2] * sin[3] + cos[0] * sin[2] * sin[3] + sin[0] * sin[1] * cos[3]) * (-sin[5] * a[6] * cos[6] - cos[5] * a[6] * sin[6] - a[5] * sin[5]) + \
        sin[0] * cos[1] * (cos[2] * cos[3] * a[3] + a[2] * cos[2]) + cos[0] * (sin[2] * cos[3] * a[3] + a[2] * sin[2]) - \
        sin[0] * sin[1] * sin[3] * a[3] + sin[0] * cos[1] * a[1]
    
    Mz =(-sin[1] * cos[2] * cos[3] - cos[1] * sin[3]) * (cos[4] * cos[5] * a[6] * cos[6] - cos[4] * sin[5] * a[6] * sin[6] - sin[4] * d[6] + cos[4] * cos[5] * a[5] + a[4] * cos[4]) + \
        sin[1] * sin[2] * (sin[4] * cos[5] * a[6] * cos[6] - sin[4] * sin[5] * a[6] * sin[6] + cos[4] * d[6] + sin[4] * cos[5] * a[5] + sin[4] * a[4]) + (-sin[1] * cos[2] * sin[3] + cos[1] * cos[3]) * \
        (-sin[5] * a[6] * cos[6] - cos[5] * a[6] * sin[6] - sin[5] * a[5]) - sin[1] * (cos[2] * cos[3] * a[3] + a[2] * cos[2]) - \
        cos[1] * sin[3] * a[3] - sin[1] * a[1] + d[0]
    
    return np.array([Mx, My, Mz])

#適応度関数
def fit(x):
    now = func(x)
    
    cm = (desire - now) 
    
    error = np.sqrt(np.sum((cm) **2))
    
    return error

#適応度関数
def fit2(x):
    global ite
    
    y = []
    for i in range(len(x)):
        desire = func(target)
        now = func2(x[i])
        
        cm = (desire - now) * 100 
        
        error = np.sqrt(np.sum((cm) **2))
        y.append(error)
    
    return y

#誤差を計算する関数
def cul_offline():
    
    #最良値を求める
    best_value = 100
    for j in range(len(x)):
        pop = x[j][:dim]
        if best_value > fit(pop):
            best_value = fit(pop)
    
    error.append(best_value)
    
    best_value = 100
    for j in range(len(x2)):
        pop = x2[j][:dim]
        if best_value > fit(pop):
            best_value = fit(pop)
    
    #誤差を記録
    error2.append(best_value)

#解を定義域に留める関数
def x_check(x):
    for j in range(dim):
        if x[j] > xmax[j]:
            x[j] = xmax[j]
        elif x[j] < xmin[j]:
            x[j] = xmin[j]
    
    return x

# ルーレット選択用関数
def roulette_choice(w):
    t = np.cumsum(w)
    r = np.random.random() * np.sum(w)  #0∼1の乱数*重みの累積和（最大値が累積和になるようにするため）
    for i, e in enumerate(t):   #インデックスと値をenumurateで取得できる間ずっと代入し続ける
        if r < e:
            return i    #累積和が乱数より大きい場合iを返す

# ABCアルゴリズム
def ABC(x,n):
    
    # employee bee step
    for i in range(n):
        v = x.copy()
        k = i
        while k == i:
            k = np.random.randint(n)
        
        r = np.random.rand()*2-1 #-1から1までの一様乱数
        j = np.random.randint(dim) #変更する次元
        v[i,j] = x[i,j] + r * (x[i,j] - x[k,j]) #近傍点計算
        v[i] = x_check(v[i])
        if fit(x[i]) > fit(v[i]):
            x[i] = v[i]
            x[i,-1] = 0
        else: x[i,-1] += 1

    # onlooker bee step
    for i in range(n):
        v = x.copy()
        w = []
        for j in range(n):
            w.append(1 / (1 + fit(x[j])))
        l = roulette_choice(w)
        
        k = l
        while k == l:
            k = np.random.randint(n)
        
        r = np.random.rand()*2-1 #-1から1までの一様乱数
        j = np.random.randint(dim)
        v[l,j] = x[l,j] + r * (x[l,j] - x[k,j])
        v[l] = x_check(v[l])
        if fit(x[l]) > fit(v[l]):
            x[l] = v[l]
            x[l,-1] = 0
        else: x[l,-1] += 1

    # scout bee step
    i = np.argmax(x,axis = 0)[-1]
    if x[i,-1] > lim:
        for j in range(dim):
            x[i][j] = (xmax[j] - xmin[j]) * np.random.rand() + xmin[j]
        x[i,-1] = 0

    return x

#探索を開始するプログラム
it = 1 
m = 2 #群れの数
update = True #最初の分割用
state = 1 #現在の変化状況
desire = func(target)

error = []    # 誤差リスト
error2 = []
x = np.zeros((s,dim))
for i in range(s):
    for j in range(dim):
        x[i][j] = (xmax[j] - xmin[j]) * np.random.rand() + xmin[j]

x = np.concatenate([x,np.zeros((s,1))],axis = 1)
x2 = x.copy()

min_scores = []

x_best = x[0].copy()
best_score = fit(x_best)

"""
# 最適化関数の重み(hyperparameter)を決める
options = {"c1": c1, "c2": c2, "w":w}

bounds = ((-180, -90, -90, -90, -90, -90, -30),(180, 30, 120, 90, 90, 90, 90))

# PSO を実行するインスタンスを設定する
optimizer = ps.single.GlobalBestPSO(n_particles = 100, dimensions = 7, options = options, bounds = bounds)

# Perform optimization
cost, pos = optimizer.optimize(fit2, iters = 100)

# 最適化された値を表示
print(cost)

# 最適化された座標を表示
print(pos)
print(func(target))
print(func(pos))

for i in range(100):
    x = ABC(x)
"""

while it < MaxIT : #itが繰り返し回数以下とbestが最大未達の時実行

    it += 1

    if update:
        sub = int(s / m) # サブ集合数
        if sub * m > len(x): # 個体数が割り切れないとき
            new = np.zeros(((sub * m) - len(x),dim + 1)) #足りない分追加
            for i in range(len(new)): #個体数調整
                new[i][:dim] = (xmax-xmin)*np.random.rand(dim) + xmin #解の初期化
                new[i][-1] = 0
            x = np.append(x,new,axis = 0)
                
        else:
            # 個体数調整
            while len(x) > s:
                x = np.delete(x,0,axis = 0)
        
        #best_before 定義
        best_before = fit(x[0][:dim])
        for i in range(s):
            pop = x[i][:dim]
            if best_before > fit(pop):
                best_before = fit(pop)
                x_best = pop.copy()
        
        #フラグ変更
        update = False
    
    x2 = ABC(x2,s)
    
    # サブポピュレーションでサーチ
    for i in range(m):
        x[i * sub:(i + 1) * sub] = ABC(x[i * sub:(i + 1) * sub],sub)#サブ集合に分割

        #fitness evaluation
        population = x[i * sub:(i + 1) * sub]
        for j in range(len(population)):
            pop = population[j][:dim]
            if best_before > fit(pop):
                best_before = fit(pop)
                x_best = pop.copy()
        
        #関数を変化させる
        if it > state * frequency:
            cul_offline()
            sheta = 2 * np.pi * ( it /MaxIT)
            desire = func(target) + [np.sin(sheta), np.cos(sheta), 0]
            state += 1
            update = True
            break
    
    # 世代進展処理    
    if update:
        
        #best_after定義
        best_after = fit(x[0][:dim])
        for i in range(s):
            pop = x[i][:dim]
            if best_after > fit(pop):
                best_after = fit(pop)
                x_best = pop.copy()
        
        #csを
        cs = best_before - best_after
        
        if cs < Tv: #定数と比較してサブ集合数を変更
            m += 1
        elif m > 2:
            m -= 1
        
        
        # クリアリングスキーム
        unique = [] #ユニークな要素があるリスト
        unique_level = 0
        #print(f"ユニーク{len(np.unique(np.round(x[:,:d],unique_level),axis = 0))}")
        for u in np.unique(np.round(x[:,:dim],unique_level),axis = 0): #ユニークな要素を検索
            unique.append(list(zip(*np.where(np.round(x[:,:dim],unique_level) == u))))
        new_x = np.zeros((len(unique),dim + 1)) #新しい母集団を定義
        for i in range(len(unique)): #ユニークな母集団を定義
            new_x[i] = x[unique[i][0][0]]

        x = new_x.copy() #母集団を定義
        if len(x) < s:
            new = np.zeros((s - len(x),dim + 1))
            for i in range(len(new)):
                for j in range(dim):
                    new[i][j] = (xmax[j] - xmin[j]) * np.random.rand() + xmin[j]
                new[i][-1] = 0
                x = np.append(x,new,axis = 0) #解の置き直し
        
        #解をシャッフルする
        np.random.shuffle(x)

#実験出力
print(it)
print(desire)
print(x_best)
print(func(x_best))
print(fit(x_best) * 100)
print(f"Multi エラー{np.round(error,4)}です")
print(f"ABC エラー{np.round(error2,4)}です")
print(f"群れの数は、{m}です")