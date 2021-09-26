# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 00:26:34 2021

@author: kazu

このプログラムは、動的最適化問題（moving peaks benchmark）を解く Multi Population ABC/
プログラムです。多群に分割して探索領域全体に分散し、探索します。
"""

#ライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import warnings

# jitのwarning用
warnings.simplefilter('ignore')

#乱数の初期値固定
np.random.seed(3)

# パラメータ設定
N = 60 # 個体数
d = 5 # 次元

lim = 30 #scout bee limit
xmax = 100 #探索範囲最大値
xmin = 0 #探索範囲最小値
frequency = 5 * 1000 #問題変化頻度
MaxIT = 50000 # 繰り返す周期
Tv = 0.05 #閾値

# the moving peak benchmark
it = 0 # 評価値関数
m = 2 #群れの数
# the moving peak benchmark
p = 20 #頂点の個数
hmax = 70 #頂点高さ最大値
hmin = 30 #頂点高さ最小値
wmax = 12 #頂点幅最大値
wmin = 1 #頂点幅最小値
hs = 7 #高さの変化強度
ws = 1 #幅の変化強度
S = 1 #移動距離
ramda = 0 #共通定数
H = 50 * np.ones(p) #高さの初期化
W = np.ones(p) #幅の初期化
F = np.random.rand(p)
v = 0
X = (xmax - xmin)*np.random.rand(p,d) + xmin  
error = []    # 誤差リスト

@jit
# ルーレット選択用関数
def roulette_choice(w):
    tot = []
    acc = 0
    for e in w:
        acc += e
        tot.append(acc)

    r = np.random.random() * acc
    for i, e in enumerate(tot):
        if r <= e:
            return i


# the moving peak
def func(x):

    #　グローバル変数導入
    global it,v,H,W,X,F

    # 5000千回ごとに変化
    if it % frequency == 0:
        # 頂点の移動 
        r = np.random.rand(p,d)
        v = S * ((1 - ramda) * r + ramda * v) / np.abs(r + v)
        #v = 0
        X = X + v
        H = H + hs * np.random.randn(p)
        H[H > 70] = 70
        H[H < 30] = 30
        W = W + ws * np.random.randn(p)
        W[W > 12] = 12
        W[W < 1] = 1
        #print(it)
        it += 1
        
    #　関数値
    F = H - W * np.sqrt(np.sum((np.tile(x,[p,1]) - X) ** 2,axis = 1))
    
    f = np.max(F)
        
    return f

# 頂点を捉えることが出来ている確認関数
def peak_check(x):
    
    #　グローバル変数導入
    global it,v,H,W,X,F
    
    index = np.argmax(H)
    distance = np.round((np.sqrt(np.sum((X[index] - x)**2)) / ((xmax - xmin) * d)),4)
    
    #　関数値
    F = H - W * np.sqrt(np.sum((np.tile(x,[p,1]) - X) ** 2,axis = 1))
    
    #　最良値の頂点番号と最大値の頂点番号
    f = np.argmax(F)
    if f == index:
        return distance,True
    else:
        return distance,False

#適応度関数
@jit
def fit(x):
    z = np.max(H) - func(x[:d])
    
    if z > 0:
        z = 1 / (1 + z)
    else:
        z = 1 + abs(z)
        
    return z

#個体の探索範囲内に戻す関数
@jit
def x_check(x):
    x[x > xmax] = xmax
    x[x < xmin] = xmin
    
    return x

# ABCアルゴリズム
def ABC(x,n):
    
    global it
    state = int(it / frequency) #現在の状態
    change = False #状況が変わったかどうか
    
    # employee bee step
    for i in range(n):
        v = x.copy()
        k = i
        while k == i:
            k = np.random.randint(n)
        
        r = np.random.rand()*2-1 #-1から1までの一様乱数
        j = np.random.randint(d) #変更する次元
        v[i,j] = x[i,j] + r * (x[i,j] - x[k,j]) #近傍点計算
        v[i] = x_check(v[i])
        it += 1 # fitness evaluation
        if fit(x[i]) < fit(v[i]):
            x[i] = v[i]
            x[i][-1] = 0
        else: x[i][-1] += 1

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
        v[l] = x_check(v[l])
        it += 1 #fitness evaluation
        if fit(x[l]) < fit(v[l]):
            x[l] = v[l]
            x[l][-1] = 0
        else: x[l][-1] += 1

    # scout bee step
    i = np.argmax(x,axis = 0)[-1]
    if x[i][-1] > lim:
        x[i][:d] = np.random.rand(d)*(xmax-xmin) + xmin
        x[i][-1] = 0

    if state != int(it / frequency): #状況が変わった時は
        chenge = True
        return x,chenge

    return x,change

#メインプログラム

# 結果出力用リスト
offline_error = []
optimum_dis = []


x = (xmax-xmin)*np.random.rand(N,d) + xmin #解の初期化
x = np.concatenate([x,np.zeros((N,1))],axis = 1)
x_best = x[0][:d] #x_bestの初期化

best_before = func(x_best) #best_after 定義
best_ratio = np.empty(0)
# アルゴリズム
while it < MaxIT : #itが繰り返し回数以下とbestが最大未達の時実行

    sub = int(N / m) # サブ集合数
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
    
    # 距離の割合計算
    distance,peak = peak_check(x_best)
    
    # サブポピュレーションでサーチ
    for i in range(m):
        x[i * sub:(i + 1) * sub],update = ABC(x[i * sub:(i + 1) * sub],sub)#サブ集合に分割
        if update == True: #環境が変化したとき
            break
        else:
            best_before = 0
            for j in range(len(x)):
                a = x[j][:d]
                if best_before < func(a):
                    best_before = func(a)
                    x_best = a.copy()
            
            off = np.max(H) - best_before
    # 世代進展処理    
    if update == True:
        # 前の変化の記録
        best_ratio = np.append(best_ratio,distance) #最高評価関数値更新
        
        # オフラインエラーを更新
        error.append(off)
        
        cs = best_before - func(x_best)
        
        if cs > Tv: #定数と比較してサブ集合数を変更
            m += 1
        elif m > 2:
            m -= 1
            
        # クリアリングスキーム
        unique = [] #ユニークな要素があるリスト
        unique_level = 2 #ユニークスキームを聞かせる程度
        for u in np.unique(np.round(x[:,:d],unique_level),axis = 0): #ユニークな要素を検索
            unique.append(list(zip(*np.where(np.round(x[:,:d],unique_level) == u))))
        new_x = np.zeros((len(unique),d + 1)) #新しい母集団を定義
        for i in range(len(unique)): #ユニークな母集団を定義
            new_x[i] = x[unique[i][0][0]]

        x = new_x.copy() #母集団を定義
        #print(f"スキーム{len(x)}")
        if len(x) < N:
            new = np.zeros((N - len(x),d + 1))
            for i in range(len(new)):
                new[i][:d] = (xmax-xmin)*np.random.rand(d) + xmin
                new[i][-1] = 0
                x = np.append(x,new,axis = 0) #解の置き直し
                
        np.random.shuffle(x) #リストをシャッフルしてランダム化
        
        # 最後の変化の記録
        if it >= MaxIT:
            # 距離の割合計算
            distance,peak = peak_check(x_best)
            
            best_ratio = np.append(best_ratio,distance) #最高評価関数値更新
        
        update = False #フラグを元に戻す

    
# 出力
mean_ratio = np.mean(best_ratio)
print(f"最大値との平均距離は{mean_ratio}です")
print(f"適応度評価は、{it}回です")   
print(f"最良解は、{x_best}です")
print(f"最良値は、{best_before}です")
print(f"最適値は、{np.max(H)}です")
print(f"エラー{np.round(error,4)}です")
print(f"群れの数は、{m}です")
plt.plot(best_ratio)
plt.xlabel("change_times")
plt.ylabel("distance_from_optimum")
plt.show()