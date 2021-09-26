"""
このプログラムは、すごろくゲームです。
自分がイベントのある位置とそのイベントを定義して改造しました。

"""
#ライブラリのインポート
import random

#定数の定義
pl_pos = 1 #プレーヤー駒の位置を表す定数
com_pos = 1 #コンピューターの駒の位置を表す定数

#改造ポイント------------
#駒の休みフラグ
pl_rest = 0
com_rest = 0

#イベント位置の定義
forward_pos1 = 5
rest_pos1 = 10
back_pos1 = 15
rest_pos2 = 25

#盤面の表示
def banmen():
    #盤面の表示
    ban = str("・"*(forward_pos1-1) + "前" + "・"*(rest_pos1-forward_pos1-1) + "休" + "・"*(back_pos1-rest_pos1-1) + "戻" + "・"*(rest_pos2-back_pos1-1
                                                                                                                           ) + "休" + "・"*(30-rest_pos2))
    #駒の位置を上書きで表示させる
    print(ban[:pl_pos-1] + "P" + ban[pl_pos:])
    print(ban[:com_pos-1] + "C" + ban[com_pos:])
#--------------------


#すごろくプログラム
banmen()
print("スゴロク、スタート！")
#ループ開始
while True:
    
    if pl_rest == 0:
        input("Enterを押すとあなたのコマが進みます")
        pl_pos = pl_pos + random.randint(1,6)
        #改造ポイント、イベントの定義------------
        if pl_pos == rest_pos1:
            print("あなたは1回お休み")
            pl_rest = 1
        if pl_pos == rest_pos2:
            print("あなたは1回お休み")
            pl_rest = 1
        if pl_pos == forward_pos1:
            print("あなたはマス進んだ！！")
            pl_pos += 2
        if pl_pos == back_pos1:
            print("あなたは2マス戻った！")
            pl_pos -= 2
        if pl_pos > 30:
            pl_pos = 30
        banmen()
        if pl_pos == 30:
            print("あなたの勝ちです！")
            break
    else:
        pl_rest = 0
    
    
    if com_rest == 0:
        input("Enterを押すとコンピュータのコマが進みます")
        com_pos = com_pos + random.randint(1,6)
        if com_pos == rest_pos1:
            print("コンピューターは1回お休み")
            com_rest = 1
        if com_pos == rest_pos2:
            print("コンピューターは1回お休み")
            com_rest = 1
        if com_pos == forward_pos1:
            print("コンピューターは2マス進んだ！！")
            com_pos += 2
        if com_pos == back_pos1:
            print("コンピューターは2マス戻った！")
            com_pos -= 2
        if com_pos > 30:
            com_pos = 30
        banmen()
        if com_pos == 30:
            print("コンピュータの勝ちです！")
            break
    else:
        com_rest = 0
        
    #-------------------------
